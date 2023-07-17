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

import torch
import os
import time
import openslide
import torchvision.transforms as transforms
import submitit
import argparse
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader

# Check the device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define validation transforms
trnsfrms_val = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


def process(bag_candidate_idx, args):
    """
    Process function to handle the extraction of patches from a bag candidate.

    Args:
        bag_candidate_idx (int): Index of the bag candidate.
        args (argparse.Namespace): Parsed command line arguments.
    """

    csvpath = os.path.join(args.output_dir, "process_list_autogen.csv")
    output_path = os.path.join(args.output_dir, "images")

    bags_dataset = Dataset_All_Bags(csvpath)
    total = len(bags_dataset)
    slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
    bag_name = slide_id+'.h5'
    h5_file_path = os.path.join(args.output_dir, 'patches', bag_name)
    slide_file_path = os.path.join(args.source_dir, slide_id+args.slide_ext)
    if os.path.isdir(slide_file_path):
        print("skip", flush=True)
        return
    print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
    print(slide_id)
    output_path_slide = os.path.join(output_path, slide_id)
    # output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
    time_start = time.time()
    wsi = openslide.open_slide(slide_file_path)
    save_patches(h5_file_path, output_path=output_path_slide,
                 wsi=wsi, target_patch_size=-1)
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
    kwargs = {'num_workers': 4,
              'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)
    mode = 'w'
    transform = transforms.ToPILImage()
    level = dataset.patch_level
    output_path = os.path.join(output_path, str(level))
    os.makedirs(output_path, exist_ok=True)
    print("tot" + str(len(dataset)))
    # Iterate over batches and save patches as images
    for count, (batch, coords) in enumerate(loader):
        print(count)
        for image, cc in zip(batch, coords):
            imagepath = os.path.join(
                output_path, "_x_"+str(int(cc[0]))+"_y_"+str(int(cc[1]))+".jpg")
            if os.path.isfile(imagepath):
                print("skip")
                continue
            image = transform(image.view(3, 256, 256))
            image.save(imagepath, quality=70)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--slide_ext', type=str, default='.tif')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)
    args = parser.parse_args()

    # Load the bags dataset through the csv file
    csvpath = os.path.join(args.output_dir, "process_list_autogen.csv")
    bags_dataset = Dataset_All_Bags(csvpath)
    total = len(bags_dataset)

    candidates = [i for i in range(total)]
    parameters = [args for i in range(total)]

    # Configure the executor for parallel execution
    executor = submitit.AutoExecutor("logs/")
    executor.update_parameters(slurm_partition="prod", name="data_prep",
                               slurm_time=600, mem_gb=15, slurm_array_parallelism=5)
    jobs = executor.map_array(process, candidates, parameters)
    # process(0,args)
