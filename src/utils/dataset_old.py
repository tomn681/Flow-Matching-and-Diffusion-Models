import os
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from skimage.transform import resize

from .data import consecutive_paths
from .utils import lot_id, load

'''
Class DefaultDataset:

Constructs a 3D DICOM-slice based dataset.
'''
class DefaultDataset(Dataset):
	
	'''
	Constructor Method
	
	Inputs:
		- basepath: (String) Annotations file directory path. 
	    
		        Directory must contain files:
		        
		          File Name:                File Column Data:
		          
		            - 'test.txt':              (Case: String, SDCT_path: String, LDCT_path: String)

		            - 'train.txt':             (Case: String, SDCT_path: String, LDCT_path: String)
		            
		           Notes: 
		            - All files must have tab-separated columns
		            - Train files include validation data
		            - _path values to images if single-image or directories if multi-file
		            
		- s_cnt: (Int) Slice count for composite images, default=3.
		            
		- img_size: (Int) Image preprocessing resize, default=512
		            
		- transforms: (object containing torchvision.transforms) Data Augmentation Transforms.
		
		- train: (Boolean) If True, opens train and validation files, default=True.
			If False opens test files and transforms = None.
			
		- load_ldct: (Boolean) If True loads LDCT as the conditioning image, default=False.
		
		- norm: (bool) Weather or not apply image normalization.
		 
		- img_datatype: (np.dtype) Data type used for image normalization, default: np.float32.
		
		- names: (list<String>) Column name identifiers. FOR INTERNAL USE. DO NOT MODIFY.
		
		- clip: NOT IMPLEMENTED VALUE
			
	Outputs:
		- dataset: (DefaultDataset Object) Dataset Object containing the given data:
			
			Attributes:
			
			 - self.img_size:	Given image preprocessing resize, default: 512
			 - self.transforms:	Given set of data augmentation transforms
			 - self.data:		(Dict) Data: Case ID, SDCT image path, LDCT image path
			 			       Keys: <Case>, <SDCT_path>, <LDCT_path>
			 			       Value Types: String
			 - self.size:		Amount of images containing in the dataset
			 - self.norm:		Weather or not apply image normalization.
			 - self.img_datatype:	Data type for normalization, if active.
			 
			Methods:
			 
			 - len:		Default len method, returns amount of images contained
			 - preprocess:		Image preprocessing and transforms applicator
			 - __getitem__: 	Default __getitem__ method for data retrieval
			 
			Note: For further details, methods are explained in it's corresponding class
	'''
	def __init__(self, file_path: str, s_cnt: int=3, img_size: int=512, norm=True, img_datatype=np.float32,
			train=True, load_ldct: bool=False, transforms=None, names=('Case','SDCT','LDCT',), clip=None,
			use_tensor_cache: bool=False, save_tensor_cache: bool=False, cache_subdir: str="_tensor_cache"):
	
		super(DefaultDataset, self).__init__()
		
		# Ensure image_size is correctly formatted
		assert img_size > 0 if img_size is not None else True, 'Size must be greater than 0 or None for full size'
			
		# Store image_size and transforms
		self.img_size = (img_size, img_size) if img_size is not None else None
		self.transforms = transforms
		self.img_datatype = img_datatype
		
		self.norm = norm
		
		self.s_cnt = s_cnt
		
		self.train = train
		self.load_ldct = bool(load_ldct)
		self.base_path = Path(file_path)
		self.split_name = "train" if train else "test"
		self.use_tensor_cache = bool(use_tensor_cache)
		self.save_tensor_cache = bool(save_tensor_cache)
		self.cache_subdir = cache_subdir
		self.names = names
		self.data_root = self._resolve_data_root()
		
		# Read train/test files
		imgs = self._read_split_file(self.data_root)
		imgs = imgs.dropna().reset_index(drop=True)
		imgs = self._expand_slices(imgs, self.data_root)
		if not imgs.empty:
			imgs = lot_id(imgs, "Case", "SDCT")	#Solo funciona si son dos o mas imágenes!
			
		# Set values
		self.data = imgs.to_dict('records')
		self.size = len(imgs)
		self.path = str(self.data_root)
		
		# Ensure not empty
		assert 0 < self.size, 'Empty Dataset'
		
		self.clip = clip
			
		# Log the dataset creation
		logging.info(f'Creating {"Train" if train else "Test"} dataset with {self.size} examples.')
		
		if self.data_root != self.base_path:
			logging.info("Dataset split %s is using tensor cache at %s", self.split_name, self.data_root)
		
	'''
	len Method
	
	Default len method. Allows to get the amount of images in dataset.
	
	Inputs: 
		- None
		
	Outputs:
		- len: (Int) Number of images in dataset
	'''
	def __len__(self):
		return self.size
		
	'''
	preprocess Method
	
	Standard preprocessing for image resizing and normalization.
	
	Inputs:
		- dcm_img: (dict) DICOM Image Dictionary.
			Keys:
				- 'Image' (np.ndarray): Loaded image.
				- 'Metadata' (dict): DICOM Metadata. If None image is used as is.
				- 'Id' (String): Image UNIQUE ID. Not shareable within study.
				
				** Metadata must include 'Rescale Slope' and 'Rescale Intercept'.
				
		- dim: (Int) Image Dimension, default=3.
	
	Outputs:
		- img_ndarray: (np.ndarray) Preprocessed image.
	'''
	def preprocess(self, dcm_img, dim: int=3, MIN_B=-1024, MAX_B=3072, slope=1.0, intersept=-1024):
		assert dim in (2,3), "Dimension dim in load() must be an integer between 2 and 3" 
		# Resize
		img_ndarray = dcm_img['Image']
		
		if dcm_img['Metadata'] is not None:
			# Rescale Factors for Hounsfield Units
			slope = float(dcm_img['Metadata']['Rescale Slope'])
			intersept = float(dcm_img['Metadata']['Rescale Intercept'])
		
		# Rescale image to Hounsfield Units
		img_ndarray = img_ndarray * slope + intersept
		
		if self.img_size and dim==3:
			img_ndarray = np.transpose(img_ndarray, (1, 2, 0))
			img_ndarray = resize(img_ndarray, self.img_size)
			img_ndarray = np.transpose(img_ndarray, (2, 0, 1))
		
		if self.norm:
			img_ndarray = (img_ndarray - MIN_B)/(MAX_B - MIN_B)
			img_ndarray = img_ndarray.astype(self.img_datatype)

		return img_ndarray
			
	
	'''
	getitem Method
	
	Default __getitem__ method. Allows iteration over the dataset.
	
	Inputs: 
		- idx: (Int) Retrieving item id.
		
	Outputs:
		- target: (dict)
			Keys:
				- image: (torch.Tensor) Low dose CT Image.
				- target: (torch.Tensor) Standard dose CT Image.
				- metadata: (dict) File metadata.
				- img_id: (String) Image ID.
				- img_path: (String) Image path. If multi-file gives central image.
				- img_size: (Int) Image size.
	'''	    
	def __getitem__(self, idx):	
		tgt = load(self.data[idx]['SDCT'], id=self.data[idx]['Case'])
		
		Id = tgt['Id']
		#metadata = tgt['Metadata']
		
		tgt = self.preprocess(tgt)
		tgt = torch.as_tensor(tgt.copy()).float().contiguous()
		
		img = None
		if not self.train or self.load_ldct:
			img = load(self.data[idx]['LDCT'], id=self.data[idx]['Case'])
			img = self.preprocess(img)
			img = torch.as_tensor(img.copy()).float().contiguous()
		
		
		# Data Augmentation
		if self.transforms is not None:
			if self.train and not self.load_ldct:
				tgt = self.transforms(tgt)
			else:
				img, tgt = self.transforms(img, tgt)
		
		if img is None:
			img = tgt
		
		# Image path
		img_path = self.data[idx]['SDCT']
		img_path = img_path[len(img_path)//2] if type(img_path)==list else img_path
		
		# Target Dictionary
		target = {}
		target['image'] = img if img is not None else None
		target['target'] = tgt
		#target['metadata'] = metadata
		target['img_id'] = Id
		target['img_path'] =  img_path
		target['img_size'] = self.img_size
		
		return target

	def _read_split_file(self, root_path: Path, names=None):
		'''
		_read_split_file Method

		Loads the annotation file (train/test) for the selected split.

		Inputs:
			- root_path: (Path) Base directory where annotation files live.
			- names: (tuple<String>) Column names (optional override).

		Outputs:
			- df: (pd.DataFrame) Parsed annotations.
		'''
		if names is None:
			names = self.names
		file_name = 'train.txt' if self.train else 'test.txt'
		target_file = root_path / file_name
		if not target_file.exists():
			raise FileNotFoundError(f"Annotations file not found: {target_file}")
		return pd.read_csv(target_file, sep='\t', names=names)

	def _expand_slices(self, df: pd.DataFrame, root_path: Path) -> pd.DataFrame:
		'''
		_expand_slices Method

		Expands directory entries into individual slice windows.

		Inputs:
			- df: (pd.DataFrame) Raw annotations.
			- root_path: (Path) Base directory used to resolve relative paths.

		Outputs:
			- expanded_df: (pd.DataFrame) Row per SDCT/LDCT window.
		'''
		records = []
		for _, row in df.iterrows():
			sdct_options = self._resolve_entry(root_path, row["SDCT"])
			ldct_options = self._resolve_entry(root_path, row["LDCT"])
			if len(sdct_options) != len(ldct_options):
				logging.warning("Skipping case %s due to mismatched slice counts (SDCT=%d, LDCT=%d)",
					row["Case"], len(sdct_options), len(ldct_options))
				continue
			for sdct_paths, ldct_paths in zip(sdct_options, ldct_options):
				records.append({
					"Case": row["Case"],
					"SDCT": self._maybe_unwrap(sdct_paths),
					"LDCT": self._maybe_unwrap(ldct_paths),
				})
		return pd.DataFrame(records)

	def _resolve_entry(self, root_path: Path, entry) -> list:
		'''
		_resolve_entry Method

		Determines the list of slice windows for the provided path.

		Inputs:
			- root_path: (Path) Base directory.
			- entry: (str) Relative or absolute path.

		Outputs:
			- windows: (list<list<String>>) Paths per window.
		'''
		full_path = self._absolute_path(root_path, entry)
		if full_path.is_dir():
			splits = consecutive_paths(str(full_path), self.s_cnt)
			return [paths for paths in splits if paths]
		return [[str(full_path)]]

	def _absolute_path(self, root_path: Path, entry) -> Path:
		'''
		_absolute_path Method

		Resolve a path relative to the dataset root.

		Inputs:
			- root_path: (Path) Dataset root path.
			- entry: (str) Entry path from annotations.

		Outputs:
			- resolved: (Path) Absolute path.
		'''
		entry_path = Path(str(entry))
		return entry_path if entry_path.is_absolute() else root_path / entry_path

	def _maybe_unwrap(self, paths):
		'''
		_maybe_unwrap Method

		Returns single-element lists as a scalar for convenience.

		Inputs:
			- paths: (list<String> | tuple<String>) Path container.

		Outputs:
			- item_or_list: (String | list<String>) Simplified container.
		'''
		if isinstance(paths, (list, tuple)) and len(paths) == 1:
			return paths[0]
		return paths

	def _resolve_data_root(self) -> Path:
		'''
		_resolve_data_root Method

		Determines whether cached data should be used or rebuilt.

		Inputs:
			- None (uses instance attributes).

		Outputs:
			- root: (Path) Active data directory.
		'''
		if not self.use_tensor_cache:
			return self.base_path
		cache_root = self.base_path / self.cache_subdir
		split_dir = cache_root / self.split_name
		if split_dir.exists():
			return split_dir
		if not self.save_tensor_cache:
			return self.base_path
		self._build_tensor_cache(split_dir)
		return split_dir if split_dir.exists() else self.base_path

	def _build_tensor_cache(self, split_dir: Path):
		'''
		_build_tensor_cache Method

		Materializes tensor caches for SDCT/LDCT splits.

		Inputs:
			- split_dir: (Path) Destination directory for cached data.

		Outputs:
			- None (writes files to disk).
		'''
		logging.info("Building tensor cache for %s split at %s", self.split_name, split_dir)
		df = self._read_split_file(self.base_path)
		df = df.dropna().reset_index(drop=True)
		if df.empty:
			logging.warning("Tensor cache skipped: no samples found for %s split.", self.split_name)
			return

		split_dir.mkdir(parents=True, exist_ok=True)
		data_dir = split_dir / "data"
		data_dir.mkdir(exist_ok=True)

		cached_entries = []
		global_idx = 0
		for row in df.to_dict('records'):
			case_id = row["Case"]
			sdct_payloads = self._generate_cache_payloads(row["SDCT"], case_id)
			ldct_payloads = self._generate_cache_payloads(row["LDCT"], case_id)
			if len(sdct_payloads) != len(ldct_payloads):
				logging.warning(
					"Skipping case %s due to mismatched cache payloads (SDCT=%d, LDCT=%d)",
					case_id, len(sdct_payloads), len(ldct_payloads)
				)
				continue

			for sdct_payload, ldct_payload in zip(sdct_payloads, ldct_payloads):
				sdct_dest = self._write_tensor_payload(sdct_payload, case_id, data_dir, global_idx, "sdct")
				ldct_dest = self._write_tensor_payload(ldct_payload, case_id, data_dir, global_idx, "ldct")
				cached_entries.append({
					"Case": case_id,
					"SDCT": os.path.relpath(sdct_dest, split_dir),
					"LDCT": os.path.relpath(ldct_dest, split_dir),
				})
				global_idx += 1

		if not cached_entries:
			logging.warning("Tensor cache skipped: no payloads generated for %s split.", self.split_name)
			return

		cache_df = pd.DataFrame(cached_entries)
		cache_file = split_dir / f"{self.split_name}.txt"
		cache_df.to_csv(cache_file, sep='\t', index=False, header=False)
		logging.info("Tensor cache saved at %s", cache_file)

	def _generate_cache_payloads(self, entry, case_id):
		'''
		_generate_cache_payloads Method

		Creates tensor payloads for a single annotation entry.

		Inputs:
			- entry: (String) SDCT/LDCT entry path.
			- case_id: (String) Case identifier.

		Outputs:
			- payloads: (list<dict>) Loaded windows ready for serialization.
		'''
		full_path = self._absolute_path(self.base_path, entry)
		if full_path.is_dir():
			splits = consecutive_paths(str(full_path), self.s_cnt)
			payloads = []
			for paths in splits:
				if not paths:
					continue
				payloads.append(load(paths, id=case_id))
			return payloads

		payload = load(str(full_path), id=case_id)
		return self._split_volume_payload(payload)

	def _split_volume_payload(self, payload):
		'''
		_split_volume_payload Method

		Slices a multi-slice payload into sequential windows.

		Inputs:
			- payload: (dict) Dictionary produced by load().

		Outputs:
			- windows: (list<dict>) Windowed payloads.
		'''
		image = payload["Image"]
		if image is None:
			return []

		if isinstance(image, torch.Tensor):
			depth = image.size(0) if image.dim() >= 3 else 1
		else:
			array = np.asarray(image)
			depth = array.shape[0] if array.ndim >= 3 else 1

		window = self.s_cnt if self.s_cnt > 0 else depth
		if depth < window:
			logging.warning("Skipping payload %s: depth %d < window %d", payload.get("Id"), depth, window)
			return []

		if depth == 1:
			return [payload]

		windows = []
		for start in range(0, depth - window + 1):
			window_payload = {
				"Image": self._slice_volume(image, start, start + window),
				"Metadata": payload.get("Metadata"),
				"Id": f"{payload.get('Id')}_S{start}",
			}
			windows.append(window_payload)
		return windows

	def _slice_volume(self, volume, start: int, end: int):
		'''
		_slice_volume Method

		Extracts a slice window from a tensor or NumPy volume.

		Inputs:
			- volume: (torch.Tensor | np.ndarray) 3D volume.
			- start: (int) Starting index (inclusive).
			- end: (int) Ending index (exclusive).

		Outputs:
			- window: (torch.Tensor | np.ndarray) Extracted block.
		'''
		if isinstance(volume, torch.Tensor):
			return volume[start:end].clone()
		array = np.asarray(volume)
		return array[start:end].copy()

	def _write_tensor_payload(self, payload, case_id, dest_dir: Path, idx: int, tag: str) -> Path:
		'''
		_write_tensor_payload Method

		Persists a payload dictionary as a .pt file.

		Inputs:
			- payload: (dict) Window payload.
			- case_id: (String) Case identifier for naming.
			- dest_dir: (Path) Output directory.
			- idx: (int) Running index for uniqueness.
			- tag: (String) Modality tag (sdct/ldct).

		Outputs:
			- dest_path: (Path) File path of saved payload.
		'''
		safe_case = str(case_id).replace(os.sep, '_')
		file_name = f"{safe_case}_{idx:06d}_{tag}.pt"
		dest_path = dest_dir / file_name
		torch.save(payload, dest_path)
		return dest_path
		
'''
Class CombinationDataset:

Constructs a 3D DICOM-slice + Sinogram dataset.
'''		
class CombinationDataset(DefaultDataset):		
	'''
	Constructor Method
	
	Inputs:
		- basepath: (String) Annotations file directory path. 
	    
		        Directory must contain files:
		        
		          File Name:                File Column Data:
		          
		            - 'test.txt':              (Case, SDCT_path, LDCT_path, SDRAW_path, LDRAW_path)

		            - 'train.txt':             (Case, SDCT_path, LDCT_path, SDRAW_path, LDRAW_path)
		            
		           Notes: 
		            - All files must have tab-separated columns
		            - Train files include validation data
		            - _path values to image if single-image or directories if multi-file
		            
		- s_cnt: (Int) Slice count for composite images, default=3.
		            
		- img_size: (Int) Image preprocessing resize, default=512
		            
		- transforms: (object containing torchvision.transforms) Data Augmentation Transforms.
		
		- train: (Boolean) If True, opens train and validation files, default=True.
			If 'test' opens test files and transforms = None.
			
		
		- norm: (bool) Wether or not apply normalization to input image to range 0-255.
		 
		- img_datatype: (np.dtype) Data type used for image normalization, default: np.uint8.
			
	Outputs:
		- dataset: (DefaultDataset Object) Dataset Object containing the given data:
			
			Attributes:
			
			 - self.img_size:	Given image preprocessing resize, default: 512
			 - self.transforms:	Given set of data augmentation transforms
			 - self.data:		(Dict) Data: Case ID, SDCT image path, LDCT image path,
			 					SDRAW image path, LDRAW image path
			 			       Keys: <Case>, <SDCT_path>, <LDCT_path>, 
			 			       	<SDRAW_path>, <LDRAW_path>
			 			       Value Types: String
			 - self.size:		Amount of images containing in the dataset
			 - self.img_datatype:	Data type for normalization, if active.
			 
			Methods:
			 
			 - len:		Default len method, returns amount of images contained
			 - preprocess:		Image preprocessing and transforms applicator
			 - __getitem__: 	Default __getitem__ method for data retrieval
			 
			Note: For further details, methods are explained in it's corresponding class
	'''
	def __init__(self, file_path: str, s_cnt: int=3, img_size: int=512, norm=True,
		img_datatype=np.float32, train=True, load_ldct: bool=False, transforms=None,
		names=('Case','SDCT','LDCT','SDRAW','LDRAW'), clip=None):
	
		super(CombinationDataset, self).__init__(
			file_path,
			s_cnt=s_cnt,
			img_size=img_size,
			norm=norm,
			img_datatype=img_datatype,
			train=train,
			load_ldct=load_ldct,
			transforms=transforms,
			names=names,
			clip=clip,
		)

	'''
	getitem Method
	
	Default __getitem__ method. Allows iteration over the dataset.
	
	Inputs: 
		- idx: (Int) Retrieving item id.
		
	Outputs:
		- target: (dict)
			Keys:
				- image: (torch.Tensor) Low dose CT Image.
				- target: (torch.Tensor) Standard dose CT Image.
				- metadata: (dict) File metadata.
				- img_id: (String) Image ID.
				- img_path: (String) Image path. If multi-file gives central image.
				- img_size: (Int) Image size.
				- sinogram: (torch.Tensor) Low Dose Sinogram.
				- tgt_sinogram: (torch.Tensor) Standard Dose Sinogram.
	'''	    
	def __getitem__(self, idx):	
		target = super(CombinationDataset, self).__getitem__(idx)
		
		tgt = load(self.data[idx]['SDRAW'], dim=2)		
		tgt = self.preprocess(tgt['Image'], dim=2)
		tgt = torch.as_tensor(tgt.copy()).float().contiguous()
		
		if not self.train or self.load_ldct:
			img = load(self.data[idx]['LDRAW'], id=self.data[idx]['Case'], dim=2)
			img = self.preprocess(img['Image'], dim=2)
			img = torch.as_tensor(img.copy()).float().contiguous()
		
		target['sinogram'] = img if not self.train and self.load_ldct else []
		target['tgt_sinogram'] = tgt
		
		return target

if __name__ == '__main__':

	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('TkAgg')

	dataset_dict = {'DefaultDataset': DefaultDataset, 
			'CombinationDataset': CombinationDataset}
	
	for dataset_name in dataset_dict.keys():
	
		print('-'*30)
		print(f'\nTesting {dataset_name}:\n')
		
		path = os.path.join('../', dataset_name)

		dataset = dataset_dict[dataset_name](path, s_cnt=1, norm=True)
		#dataset = dataset_dict[dataset_name]('./test_imgs/')
		
		#print(dataset.getinfo())
		
		tgt = dataset[30]
		img = tgt['target']
		
		plt.imshow(img.permute(1,2,0), cmap='Greys_r')
		plt.show()
		
		print(img.shape)
		
		print(f'\nFinished testing {dataset_name}:\n')
	
