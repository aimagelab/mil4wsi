# Copyright 2023 Bontempo Gianpaolo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.utils.data import DataLoader,Dataset, sampler
import argparse
import submitit
import torchvision.transforms as transforms
import openslide
import time
import os
import torch
import pandas as pd
import sys
sys.path.insert(0, '/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM/datasets')
import numpy as np
import math
import re
import pdb
import pickle
from torchvision import utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

#sys.path.append('/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM/datasets')
#from dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP


# Check the device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define validation transforms
trnsfrms_val = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]

def process(bag_candidate_idx, args):
    """
    Process function to handle the extraction of patches from a bag candidate.

    Args:
        bag_candidate_idx (int): Index of the bag candidate.
        args (argparse.Namespace): Parsed command line arguments.
    """

    #csvpath = os.path.join(args.output_dir, "prova_modifica_sort.csv")
    csvpath = os.path.join(args.output_dir, "process_list_autogen.csv")
    output_path = os.path.join(args.output_dir, "images")

    bags_dataset = Dataset_All_Bags(csvpath)
    total = len(bags_dataset)
    slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
    print(slide_id)
    bag_name = slide_id.split('/')[-1]+'.h5'
    h5_file_path = os.path.join(args.output_dir, 'patches', bag_name)
    slide_file_path = os.path.join(args.source_dir, slide_id+args.slide_ext)
    if os.path.isdir(slide_file_path):
        print("skip", flush=True)
        return
    print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
    print(slide_id)
    output_path_slide = os.path.join(output_path, slide_id.split('/')[-1])
    # output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
    time_start = time.time()
    try:
        wsi = openslide.open_slide(slide_file_path)
        save_patches(h5_file_path, output_path=output_path_slide,
                    wsi=wsi, target_patch_size=-1)
    except:
        print("No patches found at that resolution for this slide",slide_id)
        pass
    time_stop = time.time()


def save_patches(file_path, output_path, wsi, target_patch_size,
                 batch_size=160):
    """
    Function to save patches from a bag (.h5 file) and store them as images.

    Args:
        file_path (str): Directory of the bag (.h5 file).
        output_path (str): Directory to save computed features (.h5 file).
        wsi (openslide.openslide): Whole slide image (WSI) object.
        target_patch_size (int): Custom defined, rescaled image size before embedding.
        batch_size (int): Batch size for computing features in batches.
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi,
                                 target_patch_size=target_patch_size, custom_transforms=trnsfrms_val)
    x, y = dataset[0]
    kwargs = {'num_workers': 2,
              'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)
    mode = 'w'
    transform = transforms.ToPILImage()
    level = dataset.patch_level
    #output_path = os.path.join(output_path, str(level))
    os.makedirs(output_path, exist_ok=True)
    print("tot" + str(len(dataset)))
    # Iterate over batches and save patches as images
    for count, (batch, coords) in enumerate(loader):
        print(count)
        
        for image, cc in zip(batch, coords):
            imagepath = os.path.join(output_path, "_x_"+str(int(cc[0]))+"_y_"+str(int(cc[1]))+".jpg")
#                output_path, "x-"+str(int(cc[0]))+"_y-"+str(int(cc[1]))+".png")
            if os.path.isfile(imagepath):
                print("skip")
                continue
            # try:
            image = transform(image.view(3, dataset.patch_size, dataset.patch_size))
            # print("Transformed with 3,1024,1024")
            # except:
            #     try:
            #         image= transform(image.view(3, 512, 512))
            #         print("Transformed with 3,512,512")
            #     except:
            #         image= transform(image.view(3, 256, 256)) 
            #         print("Transformed with 3,256,256")
            image.save(imagepath, quality=70)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--output_dir', type=str, default="/work/H2020DeciderFicarra/fmiccolis/WP2/Oskari_framework/patches_step0_test")
    parser.add_argument('--source_dir', type=str, default="/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022")
    parser.add_argument('--slide_ext', type=str, default='.mrxs')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)
    args = parser.parse_args()
    #args.output_dir = 
    
    # "/mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard/x5"
    #args.source_dir = 
    #args.slide_ext = ".mrxs"
    # Load the bags dataset through the csv file
    csvpath = os.path.join(args.output_dir, "process_list_autogen.csv")
    #csvpath = os.path.join(args.output_dir, "prova_modifica_sort.csv")
    bags_dataset = Dataset_All_Bags(csvpath)
    total = len(bags_dataset)

    candidates = [i for i in range(total)]
    parameters = [args for i in range(total)]

    # Configure the executor for parallel execution
    executor = submitit.AutoExecutor("logs/PRINN/x20/otsu_false")
    executor.update_parameters(slurm_partition="all_usr_prod", name="conversion_jpg", slurm_additional_parameters={"account":"h2020deciderficarra"},
                               slurm_time=600, mem_gb=15, slurm_array_parallelism=5)
#    jobs = executor.map_array(process, candidates, parameters)
    
    
#    FOR DEBUG
    for j in range(0,total):
        csvpath = os.path.join(args.output_dir, "process_list_autogen.csv")
        output_path = os.path.join(args.output_dir, "images")
        bags_dataset = Dataset_All_Bags(csvpath)
        slide_id = bags_dataset[j].split(args.slide_ext)[0]
        output_path = os.path.join(args.output_dir, "images")
        if os.path.isdir(os.path.join(output_path, slide_id.split('/')[-1])):
            print(slide_id.split('/')[-1], " already done")
            continue
        else:
            process(candidates[j], args)
    


