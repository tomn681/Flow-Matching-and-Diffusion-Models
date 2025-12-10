import os
import torch
import logging
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from skimage.transform import resize

from .utils import n_slice_split, lot_id, load

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
			
		- diff: (Boolean) If True loads only target image for training, default=True.
		
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
			train=True, diff=True, transforms=None, names=('Case','SDCT','LDCT',), clip=None):
	
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
		self.diff = diff
		
		# Read train/test files
		imgs = pd.read_csv(os.path.join(file_path, 'train.txt' if train else 'test.txt'), sep='\t', \
		 names=names)
			
		imgs = imgs.dropna().reset_index(drop=True) #AQUI SE BORRAN IMAGENES QUE SI SIRVEN PARA TRAIN
		 
		if os.path.isdir(str(imgs["SDCT"][0])):
			imgs["SDCT"] = imgs["SDCT"].apply(lambda x: n_slice_split(os.path.join(file_path, x), self.s_cnt))
			imgs["LDCT"] = imgs["LDCT"].apply(lambda x: n_slice_split(os.path.join(file_path, x), self.s_cnt))
		
			imgs = imgs.explode(["SDCT", "LDCT"]).dropna().reset_index(drop=True)  #AQUI SE BORRAN IMAGENES QUE SI SIRVEN PARA TRAIN
			
			imgs = lot_id(imgs, "Case", "SDCT")	#Solo funciona si son dos o mas imágenes!
			
		# Set values
		self.data = imgs.to_dict('records')
		self.size = len(imgs)
		self.path = file_path
		
		# Ensure not empty
		assert 0 < self.size, 'Empty Dataset'
		
		self.clip = clip
			
		# Log the dataset creation
		logging.info(f'Creating {"Train" if train else "Test"} dataset with {self.size} examples.')
		
		
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
		if not self.train or not self.diff:
			img = load(self.data[idx]['LDCT'], id=self.data[idx]['Case'])
			img = self.preprocess(img)
			img = torch.as_tensor(img.copy()).float().contiguous()
		
		
		# Data Augmentation
		if self.transforms is not None:
			if self.train and self.diff:
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
		img_datatype=np.float32, train=True, transforms=None, names=('Case','SDCT','LDCT','SDRAW','LDRAW'), 
		clip=None):
	
		super(CombinationDataset, self).__init__(file_path, s_cnt, img_size, norm,
			img_datatype, train, transforms, names, clip)

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
		
		if not self.train or not self.diff:
			img = load(self.data[idx]['LDRAW'], id=self.data[idx]['Case'], dim=2)
			img = self.preprocess(img['Image'], dim=2)
			img = torch.as_tensor(img.copy()).float().contiguous()
		
		target['sinogram'] = img if not self.train and not self.diff else []
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
	
