import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from skimage.transform import resize

from utils.dataset_utils import absolute_path, cache_path_for_entry, maybe_unwrap, resolve_entry, save_tensor_cache, split_volume_entry
from utils.utils import lot_id

from .base import BaseDataset
try:
    import pydicom
    from pydicom.dataset import Dataset as DICOMDataset, FileDataset
except Exception:  # pragma: no cover - optional dependency
    pydicom = None
    DICOMDataset = None
    FileDataset = None


class LDCTDataset(BaseDataset):
    """
    LDCT/SDCT dataset. Overrides preprocess to apply HU conversion and CT normalization.
    """
    def __init__(
        self,
        file_path: str,
        train: bool = True,
        img_size: int | Tuple[int, int] | Tuple[int, int, int] | None = None,
        window_size: int = 1,
        norm: bool = True,
        img_datatype=np.float32,
        transforms=None,
        load_ldct: bool = False,
        names: Tuple[str, ...] = ("Case", "SDCT", "LDCT"),
        split_file: str | Path | None = None,
        use_tensor_cache: bool = True,
        save_tensor_cache: bool = False,
        cache_subdir: str = "cache",
    ):
        super().__init__(
            file_path=file_path,
            train=train,
            img_size=img_size,
            norm=norm,
            img_datatype=img_datatype,
            transforms=transforms,
            conditioning=load_ldct,
            id_key="Case",
            target_key=names[1],
            conditioning_key=names[2],
            split_names=names,
            split_file=split_file,
            use_tensor_cache=use_tensor_cache,
            save_tensor_cache=save_tensor_cache,
            cache_subdir=cache_subdir,
        )
        self.names = names
        self.window_size = int(window_size) if window_size is not None else 1
        self._build_ldct_index(names)

    def _build_ldct_index(self, names: Tuple[str, ...]) -> None:
        df = self._read_split_file(self.data_root, names=names)
        df = df.dropna().reset_index(drop=True)
        records = []
        for _, row in df.iterrows():
            sdct_path = absolute_path(self.data_root, row[names[1]])
            ldct_path = absolute_path(self.data_root, row[names[2]])
            sdct_opts = resolve_entry(self.data_root, row[names[1]], self.window_size) if sdct_path.is_dir() else split_volume_entry(str(sdct_path), self.window_size)
            ldct_opts = resolve_entry(self.data_root, row[names[2]], self.window_size) if ldct_path.is_dir() else split_volume_entry(str(ldct_path), self.window_size)
            if len(sdct_opts) != len(ldct_opts):
                logging.warning(
                    "Skipping case %s due to mismatched slice counts (SDCT=%d, LDCT=%d)",
                    row["Case"], len(sdct_opts), len(ldct_opts)
                )
                continue
            for sdct_idx, (sdct_paths, ldct_paths) in enumerate(zip(sdct_opts, ldct_opts)):
                sdct_entry = maybe_unwrap(sdct_paths) if isinstance(sdct_paths, (list, tuple)) else sdct_paths
                ldct_entry = maybe_unwrap(ldct_paths) if isinstance(ldct_paths, (list, tuple)) else ldct_paths
                sdct_split_idx = sdct_entry.get("split_index") if isinstance(sdct_entry, dict) else sdct_idx
                sdct_split_cnt = sdct_entry.get("split_count", len(sdct_opts)) if isinstance(sdct_entry, dict) else len(sdct_opts)
                ldct_split_idx = ldct_entry.get("split_index") if isinstance(ldct_entry, dict) else sdct_idx
                ldct_split_cnt = ldct_entry.get("split_count", len(ldct_opts)) if isinstance(ldct_entry, dict) else len(ldct_opts)
                records.append({
                    "Case": row["Case"],
                    names[1]: sdct_entry,
                    names[2]: ldct_entry,
                    f"{names[1]}__split_index": sdct_split_idx,
                    f"{names[1]}__split_count": sdct_split_cnt,
                    f"{names[2]}__split_index": ldct_split_idx,
                    f"{names[2]}__split_count": ldct_split_cnt,
                })
        if not records:
            raise ValueError("Empty Dataset")
        df = pd.DataFrame(records)
        df = lot_id(df, "Case", names[1])
        self.data = df.to_dict("records")
        self.size = len(self.data)

    def _cache_info(self, entry, row, key: str | None):
        if key is None:
            return None, 1
        return row.get(f"{key}__split_index"), row.get(f"{key}__split_count", 1)


    def preprocess(
        self,
        payload: dict,
        MIN_B: float = -1024,
        MAX_B: float = 3072,
        slope: float = 1.0,
        intersept: float = -1024,
    ) -> np.ndarray:
        img = payload["Image"] if isinstance(payload, dict) else payload
        meta = payload.get("Metadata") if isinstance(payload, dict) else None
        if meta is not None:
            try:
                slope = float(meta.get("Rescale Slope", slope))
                intersept = float(meta.get("Rescale Intercept", intersept))
            except (TypeError, ValueError):
                pass
        img = np.asarray(img) * slope + intersept
        if self.img_size is not None:
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
                img = resize(img, self.img_size, preserve_range=True)
                img = np.transpose(img, (2, 0, 1))
            else:
                img = resize(img, self.img_size, preserve_range=True)
        img = self.to_image(img, MIN_B=MIN_B, MAX_B=MAX_B)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return img.astype(self.img_datatype)

    def to_image(self, img: np.ndarray, MIN_B: float = -1024, MAX_B: float = 3072) -> np.ndarray:
        """Map HU/windowed CT data into canonical image space [0, 1]."""
        img = np.asarray(img)
        if self.norm:
            denom = (MAX_B - MIN_B) if MAX_B != MIN_B else 1.0
            img = (img - MIN_B) / denom
        return np.clip(img, 0.0, 1.0).astype(self.img_datatype)

    def from_image(self, img: np.ndarray | torch.Tensor, MIN_B: float = -1024, MAX_B: float = 3072):
        """Invert canonical image space [0, 1] back into the configured HU window."""
        scale = (MAX_B - MIN_B)
        if isinstance(img, torch.Tensor):
            return img.clamp(0.0, 1.0) * scale + MIN_B
        img = np.clip(np.asarray(img), 0.0, 1.0)
        return (img * scale + MIN_B).astype(self.img_datatype)


class LDCTAttentionDataset(LDCTDataset):
    """
    LDCT dataset that keeps conditioning tensors raw while normalizing targets.
    """
    def __getitem__(self, idx):
    def save_output(self, row: dict, key: str, tensor, output_root: Path) -> None:
        """
        LDCT-specific writer:
          - 2D image (or single-slice [1,H,W]) -> save both PNG and DICOM
          - volume ([D,H,W] or [1,D,H,W]) -> save DICOM slices
          - fallback -> .pt tensor cache
        """
        entry = row.get(key)
        split_index, split_count = self._cache_info(entry, row, key)
        out_path = cache_path_for_entry(self.base_path, output_root, entry, split_index, split_count)
        if out_path is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arr = torch.as_tensor(tensor).detach().cpu().float()
        arr_np = arr.numpy()
        source_meta = self._source_metadata(row, key)
        if arr_np.ndim == 4 and arr_np.shape[0] == 1:
            arr_np = arr_np[0]

        if arr_np.ndim == 2 or (arr_np.ndim == 3 and arr_np.shape[0] == 1):
            img2d = arr_np if arr_np.ndim == 2 else arr_np[0]
            self._save_png(img2d, out_path.with_suffix(".png"))
            self._save_dicom_slice(img2d, out_path.with_suffix(".dcm"), metadata=source_meta)
            return

        if arr_np.ndim == 3:
            # expected [D,H,W]
            vol_dir = out_path.with_suffix("")
            vol_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(arr_np.shape[0]):
                self._save_dicom_slice(arr_np[idx], vol_dir / f"slice_{idx:04d}.dcm", metadata=source_meta)
            return

        save_tensor_cache(arr, out_path)

    def _source_metadata(self, row: dict, key: str):
        """
        Retrieve original metadata from source entry when available.
        """
        entry = row.get(key)
        if entry is None:
            return None
        try:
            payload = self._load_entry(entry, row.get(self.id_key) if self.id_key else None)
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload.get("Metadata")
        return None

    @staticmethod
    def _save_png(img: np.ndarray, path: Path) -> None:
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(u8, mode="L").save(path)

    @staticmethod
    def _save_dicom_slice(img: np.ndarray, path: Path, metadata: dict | None = None) -> None:
        if pydicom is None or FileDataset is None or DICOMDataset is None:
            # fallback when pydicom is unavailable
            np.save(path.with_suffix(".npy"), np.asarray(img, dtype=np.float32))
            return

        px = np.asarray(np.clip(img, 0.0, 1.0) * 4095.0, dtype=np.uint16)
        file_meta = DICOMDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.Rows = int(px.shape[0])
        ds.Columns = int(px.shape[1])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        if metadata is not None:
            slope = metadata.get("Rescale Slope", metadata.get("RescaleSlope", 1))
            intercept = metadata.get("Rescale Intercept", metadata.get("RescaleIntercept", -1024))
        else:
            slope, intercept = 1, -1024
        ds.RescaleIntercept = float(intercept)
        ds.RescaleSlope = float(slope)
        if metadata is not None:
            thickness = LDCTDataset._meta_float(metadata, "Slice Thickness", "SliceThickness")
            spacing_between = LDCTDataset._meta_float(metadata, "Spacing Between Slices", "SpacingBetweenSlices")
            pixel_spacing = metadata.get("Pixel Spacing", metadata.get("PixelSpacing"))
            if thickness is not None:
                ds.SliceThickness = float(thickness)
            if spacing_between is not None:
                ds.SpacingBetweenSlices = float(spacing_between)
            if pixel_spacing is not None:
                if isinstance(pixel_spacing, str):
                    parts = [p for p in pixel_spacing.replace("\\", ",").split(",") if p.strip()]
                    if len(parts) >= 2:
                        ds.PixelSpacing = [str(float(parts[0])), str(float(parts[1]))]
                elif isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
                    ds.PixelSpacing = [str(float(pixel_spacing[0])), str(float(pixel_spacing[1]))]
        ds.PixelData = px.tobytes()
        ds.save_as(str(path), write_like_original=False)

    @staticmethod
    def _meta_float(meta: dict, *keys: str):
        for key in keys:
            value = meta.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None


class LDCTAttentionDataset(LDCTDataset):
    """
    LDCT dataset that skips preprocessing for conditioning inputs (e.g., VAE latents).
    """
    def _load_conditioning_tensor(self, row: dict, item_id):
        if self.conditioning_key is None:
            raise KeyError("Conditioning requested but no conditioning column provided.")
        return self._load_entry_tensor(row, item_id, self.conditioning_key, preprocess=False)




def build_ldct_from_config(training_cfg: dict, _model_cfg: dict | None, train: bool):
    """
    Factory for LDCTDataset used by the dataset registry.
    """
    data_root = Path(training_cfg["data_root"])
    img_size = training_cfg.get("img_size")
    window_size = training_cfg.get("window_size", training_cfg.get("slice_count", 1))
    load_ldct = bool(training_cfg.get("load_ldct", False))
    use_tensor_cache = bool(training_cfg.get("use_tensor_cache", True))
    save_tensor_cache = bool(training_cfg.get("save_tensor_cache", False))
    cache_subdir = training_cfg.get("tensor_cache_subdir", "cache")
    norm = training_cfg.get("norm", True)
    return LDCTDataset(
        str(data_root),
        train=train,
        img_size=img_size,
        window_size=window_size,
        norm=norm,
        load_ldct=load_ldct,
        use_tensor_cache=use_tensor_cache,
        save_tensor_cache=save_tensor_cache,
        cache_subdir=cache_subdir,
    )


def run_self_tests() -> None:
    """
    Lightweight tests for LDCTDataset slicing and preprocessing.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sdct_path = data_dir / "sdct.npy"
        ldct_path = data_dir / "ldct.npy"
        volume = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        np.save(sdct_path, volume)
        np.save(ldct_path, volume)
        (root / "train.txt").write_text("Case\tSDCT\tLDCT\nC1\tdata/sdct.npy\tdata/ldct.npy\n")

        ds = LDCTDataset(
            file_path=str(root),
            window_size=1,
            img_size=None,
            load_ldct=True,
        )
        assert len(ds) == 3, "LDCTDataset should expand each slice for window_size=1."
        sample = ds[0]
        assert sample["target"].shape[0] == 1, "LDCTDataset should add channel dimension."
        assert sample["image"] is not None, "Conditioning image should be present when load_ldct=True."
