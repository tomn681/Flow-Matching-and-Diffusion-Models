import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from .dataset_utils import (
	cache_path_for_entry,
	save_tensor_cache,
	absolute_path,
	maybe_unwrap,
	resolve_entry,
	split_volume_entry,
)
from .utils import load, lot_id


class BaseDataset(Dataset):
	"""
	Class BaseDataset:

	Constructs a generic image dataset with optional conditioning and tensor caching.
	This base class expects train/test split files with tab-separated columns and
	returns standard RGB/grayscale tensors by default.
	"""
	def __init__(
		self,
		file_path: str,
		train: bool = True,
		img_size: int | Tuple[int, int] | Tuple[int, int, int] | None = None,
		norm: bool = True,
		img_datatype=np.float32,
		transforms=None,
		conditioning: bool = False,
		id_key: str | None = None,
		target_key: str = "target",
		conditioning_key: str | None = "conditioning",
		split_names: Tuple[str, ...] | None = None,
		use_tensor_cache: bool = True,
		save_tensor_cache: bool = False,
		cache_subdir: str = "cache",
		preprocess_kwargs: dict | None = None,
	):
		"""
		Constructor Method

		Inputs:
			- file_path: (String) Root directory containing train/test split files.
			- train: (Boolean) If True uses train split, else test split.
			- img_size: (Int | Tuple | None) Resize target. If None keeps original size.
			- norm: (Boolean) If True, applies normalization (integer: dtype range, float: min/max if outside [0,1]).
			- img_datatype: (np.dtype) Output dtype after preprocessing, default np.float32.
			- transforms: (callable) Optional data augmentation transforms.
			- conditioning: (Boolean) If True loads conditioning image (second input).
			- id_key: (String | None) Column name for item id in split files.
			- target_key: (String) Column name for target image path.
			- conditioning_key: (String | None) Column name for conditioning image path.
			- split_names: (Tuple | None) Column names override for reading split files.
			- use_tensor_cache: (Boolean) If True and cache exists, read cached tensors.
			- save_tensor_cache: (Boolean) If True, save tensors to cache when missing.
			- cache_subdir: (String) Cache folder name, default "cache".
			- preprocess_kwargs: (dict | None) Extra kwargs passed to preprocess().

		Outputs:
			- dataset: (BaseDataset Object) Dataset instance with loaded metadata.
		"""
		super().__init__()
		self.base_path = Path(file_path)
		self.train = train
		self.split_name = "train" if train else "test"
		self.id_key = id_key
		self.target_key = target_key
		self.conditioning_key = conditioning_key
		self.img_size = self._normalize_img_size(img_size)
		self.norm = bool(norm)
		self.img_datatype = img_datatype
		self.transforms = transforms
		self.conditioning = bool(conditioning)
		self.use_tensor_cache = bool(use_tensor_cache) or bool(save_tensor_cache)
		self.save_tensor_cache = bool(save_tensor_cache)
		self.cache_subdir = cache_subdir
		self.cache_root = self.base_path / self.cache_subdir
		self.preprocess_kwargs = dict(preprocess_kwargs) if preprocess_kwargs else {}

		self.data_root = self.base_path
		df = self._read_split_file(self.data_root, names=split_names)
		df = df.dropna().reset_index(drop=True)
		self.data = df.to_dict("records")
		self.size = len(self.data)
		assert self.size > 0, "Empty Dataset"
		logging.info("Creating %s dataset with %d examples.", self.split_name.capitalize(), self.size)

	def _normalize_img_size(self, img_size):
		"""
		_normalize_img_size Method

		Normalizes the img_size input to a tuple.

		Inputs:
			- img_size: (Int | Tuple | None)

		Outputs:
			- img_size: (Tuple | None) Normalized size.
		"""
		if img_size is None:
			return None
		if isinstance(img_size, int):
			return (img_size, img_size)
		return tuple(img_size)

	def __len__(self):
		"""
		len Method

		Outputs:
			- len: (Int) Number of samples.
		"""
		return self.size

	def _read_split_file(self, root_path: Path, names=None):
		"""
		_read_split_file Method

		Loads the train/test split file.

		Inputs:
			- root_path: (Path) Base directory containing train/test files.
			- names: (Tuple | None) Column names override.

		Outputs:
			- df: (pd.DataFrame) Parsed split table.
		"""
		file_name = "train.txt" if self.train else "test.txt"
		target_file = root_path / file_name
		if not target_file.exists():
			raise FileNotFoundError(f"Annotations file not found: {target_file}")
		if names is None:
			return pd.read_csv(target_file, sep="\t")
		return pd.read_csv(target_file, sep="\t", names=names)

	def preprocess(self, payload: dict) -> np.ndarray:
		"""
		preprocess Method

		Standard preprocessing for generic images.
		- Resizes to img_size if provided.
		- Normalizes integer images to [0,1].
		- Normalizes float images to [0,1] if values are outside that range.

		Inputs:
			- payload: (dict) Output from load().

		Outputs:
			- img: (np.ndarray) Preprocessed image.
		"""
		img = payload["Image"] if isinstance(payload, dict) else payload
		img = np.asarray(img)
		if self.img_size is not None:
			img = resize(img, self.img_size, preserve_range=True)
		if self.norm:
			if np.issubdtype(img.dtype, np.integer):
				max_val = np.iinfo(img.dtype).max
				if max_val > 0:
					img = img / max_val
			else:
				img_min = float(np.min(img)) if img.size else 0.0
				img_max = float(np.max(img)) if img.size else 0.0
				if img_max > 1.0 or img_min < 0.0:
					denom = (img_max - img_min) if img_max != img_min else 1.0
					img = (img - img_min) / denom
		return img.astype(self.img_datatype)

	def __getitem__(self, idx):
		"""
		getitem Method

		Loads a single sample, optionally using cached tensors.
		If caching is enabled and the tensor exists in cache, it is loaded from disk.
		Otherwise it is loaded from the original path, preprocessed, and cached if requested.

		Inputs:
			- idx: (Int) Sample index.

		Outputs:
			- target: (dict) Keys: image, target, img_id, img_path, img_size
		"""
		row = self.data[idx]
		target_key = self.target_key
		cond_key = self.conditioning_key

		item_id = row.get(self.id_key) if self.id_key else None
		tgt_entry = row[target_key]
		tgt_split_index, tgt_split_count = self._cache_info(tgt_entry, row, target_key)
		tgt_cache_path = cache_path_for_entry(
			self.base_path,
			self.cache_root,
			tgt_entry,
			tgt_split_index,
			tgt_split_count,
		)
		if self.use_tensor_cache and tgt_cache_path is not None and tgt_cache_path.exists():
			tgt = torch.load(tgt_cache_path)
			tgt = torch.as_tensor(tgt).float().contiguous()
		else:
			tgt_payload = self._load_entry(tgt_entry, item_id)
			try:
				tgt = self.preprocess(tgt_payload, **self.preprocess_kwargs) if self.preprocess_kwargs else self.preprocess(tgt_payload)
			except TypeError as exc:
				raise TypeError(f"Invalid preprocess kwargs for {self.__class__.__name__}: {self.preprocess_kwargs}") from exc
			tgt = torch.as_tensor(tgt).float().contiguous()
			if self.save_tensor_cache and tgt_cache_path is not None and not tgt_cache_path.exists():
				save_tensor_cache(tgt, tgt_cache_path)

		img = None
		if self.conditioning:
			if cond_key is None:
				raise KeyError("Conditioning requested but no conditioning column provided.")
			cond_entry = row[cond_key]
			cond_split_index, cond_split_count = self._cache_info(cond_entry, row, cond_key)
			cond_cache_path = cache_path_for_entry(
				self.base_path,
				self.cache_root,
				cond_entry,
				cond_split_index,
				cond_split_count,
			)
			if self.use_tensor_cache and cond_cache_path is not None and cond_cache_path.exists():
				img = torch.load(cond_cache_path)
				img = torch.as_tensor(img).float().contiguous()
			else:
				cond_payload = self._load_entry(cond_entry, item_id)
				try:
					img = self.preprocess(cond_payload, **self.preprocess_kwargs) if self.preprocess_kwargs else self.preprocess(cond_payload)
				except TypeError as exc:
					raise TypeError(f"Invalid preprocess kwargs for {self.__class__.__name__}: {self.preprocess_kwargs}") from exc
				img = torch.as_tensor(img).float().contiguous()
				if self.save_tensor_cache and cond_cache_path is not None and not cond_cache_path.exists():
					save_tensor_cache(img, cond_cache_path)

		if self.transforms is not None:
			if self.train and not self.conditioning:
				tgt = self.transforms(tgt)
			else:
				img, tgt = self.transforms(img, tgt)

		if img is None:
			img = tgt

		target = {
			"image": img,
			"target": tgt,
			"img_id": item_id,
			"img_path": self._resolve_img_path(row.get(target_key)),
			"img_size": self.img_size,
		}
		return target

	def _resolve_img_path(self, entry):
		"""
		_resolve_img_path Method

		When a sample is a window of multiple slices, return the middle slice path
		as a representative. For a single path, return it as is.

		Inputs:
			- entry: (String | list) Image path or list of paths.

		Outputs:
			- path: (String) Representative image path.
		"""
		if isinstance(entry, list):
			return entry[len(entry) // 2]
		if isinstance(entry, dict):
			return entry.get("path")
		return entry

	def _cache_info(self, entry, row, key: str | None):
		"""
		_cache_info Method

		Provides split metadata for cache naming. Subclasses that perform splitting
		should override this to return (split_index, split_count).

		Inputs:
			- entry: (Any) Entry payload for the sample.
			- row: (dict) Row metadata.
			- key: (String | None) Column key for the entry.

		Outputs:
			- split_index: (Int | None)
			- split_count: (Int)
		"""
		return None, 1

	def _load_entry(self, entry, item_id):
		"""
		_load_entry Method

		Loads an entry that may be a single path, list of paths, or a split dict.

		Inputs:
			- entry: (String | list | dict) Entry payload.
			- item_id: (String | None) Optional id.

		Outputs:
			- payload: (dict) Image payload with Image/Metadata/Id.
		"""
		if isinstance(entry, list):
			return load(entry, id=item_id)
		if isinstance(entry, dict):
			payload = load(entry["path"], id=item_id)
			window = int(entry.get("window", 1))
			start = int(entry.get("split_index", 0))
			return self._slice_payload(payload, start, window)
		return load(entry, id=item_id)

	def _slice_payload(self, payload, start: int, window: int):
		"""
		_slice_payload Method

		Slices a payload along the first dimension for volume windows.

		Inputs:
			- payload: (dict) Image payload from load().
			- start: (Int) Window start index.
			- window: (Int) Window size.

		Outputs:
			- payload: (dict) Sliced payload.
		"""
		image = payload.get("Image") if isinstance(payload, dict) else None
		if image is None or window <= 0:
			return payload
		if isinstance(image, torch.Tensor):
			sliced = image[start : start + window].clone()
		else:
			array = np.asarray(image)
			sliced = array[start : start + window].copy()
		return {"Image": sliced, "Metadata": payload.get("Metadata"), "Id": payload.get("Id")}


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
		if self.norm:
			img = (img - MIN_B) / (MAX_B - MIN_B)
		return img.astype(self.img_datatype)


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
	Lightweight integrity and failure tests for BaseDataset caching.
	"""
	import tempfile

	if torch is None:
		raise RuntimeError("torch is required for dataset self-tests.")

	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		sample_dir = root / "data"
		sample_dir.mkdir(parents=True, exist_ok=True)
		sample_path = sample_dir / "sample.npy"
		np.save(sample_path, np.arange(6, dtype=np.float32).reshape(2, 3))
		(root / "train.txt").write_text("target\n" + "data/sample.npy\n")

		ds = BaseDataset(
			file_path=str(root),
			use_tensor_cache=True,
			save_tensor_cache=True,
			cache_subdir="cache",
		)
		first = ds[0]["target"].clone()
		cache_path = root / "cache" / "data" / "sample.pt"
		assert cache_path.exists(), "Cache file was not created."

		np.save(sample_path, np.zeros((2, 3), dtype=np.float32))
		second = ds[0]["target"]
		assert torch.equal(first, second), "Cache was not used on second access."

		ds_fail = BaseDataset(
			file_path=str(root),
			use_tensor_cache=False,
			save_tensor_cache=False,
			preprocess_kwargs={"bad_key": 1},
		)
		try:
			_ = ds_fail[0]
		except TypeError:
			pass
		else:
			raise AssertionError("Invalid preprocess kwargs did not raise TypeError.")
