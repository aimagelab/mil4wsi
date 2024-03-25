import shutil
import glob
import os
import submitit
import pandas as pd
import argparse


def calcolate_shift(patch, level, base_size):
    """
    Calculate the shift coordinates for a given patch.

    Args:
        patch (str): Path to the patch image file.
        level (int): Level of the pyramid where the patch is located.
        base_size (int): Base size of the patch at the highest resolution.

    Returns:
        tuple: Tuple containing the x-coordinate, y-coordinate, and shift value.
            - x (int): The x-coordinate of the patch.
            - y (int): The y-coordinate of the patch.
            - shift (int): The shift value for the patch at the specified level.
    """
    # Calculate the shift value based on the resolution level and base size
    shift = base_size/(2**level)
    x = int(patch.split(os.sep)[-1].split("_")[2])
    y = int(patch.split(os.sep)[-1].split("_")[4].split(".")[0])
    return x, y, int(shift)


def hierarchy(parent_path, x10path, x20path, dirpath, base_size, level,x10processed, x20processed):
    """
    Generate the hierarchical structure of directories and copy files accordingly to the resolution.

    Args:
        parent_path (str): Path to the parent file (patch).
        x10path (str): Path to the x10 resolution directory.
        x20path (str): Path to the x20 resolution directory.
        dirpath (str): Path to the current directory.
        base_size (int): Base size of the patch at the highest resolution.
        level (int): Current level of the hierarchy.
    """
    if level == 3:
        return x10processed,x20processed
    os.makedirs(dirpath, exist_ok=True)
    x, y, shift = calcolate_shift(parent_path, level, base_size)
    if level == 1:
        base = x10path
    else:
        base = x20path
    f1 = base+"_x_"+str(x)+"_y_"+str(y)+".jpg"
    f2 = base+"_x_"+str(x)+"_y_"+str(int(y+shift))+".jpg"
    f3 = base+"_x_"+str(int(x+shift))+"_y_"+str(int(y+shift))+".jpg"
    f4 = base+"_x_"+str(int(x+shift))+"_y_"+str(y)+".jpg"
    for file in [f1, f2, f3, f4]:
        if (os.path.isfile(file)):
            newname = os.path.basename(file)
            if 'x20' in file:
                x20processed.append(newname)
            if 'x10' in file:
                x10processed.append(newname)
            if (not os.path.isfile(os.path.join(dirpath, newname))):
                shutil.copy(file, os.path.join(dirpath, newname))
            tmpresx10, tmpresx20 = hierarchy(file, x10path, x20path, os.path.join(
                dirpath, newname.split(".")[0]), base_size, level+1,x10processed, x20processed)
    return x10processed,x20processed

def prepareslide(candidates, args):
    """
    Prepare slide data for processing

    Args:
        candidate (int): Index of the candidate slide.
        args (argparse.Namespace): Parsed command line arguments.
    """
    log_folder = "LOGFOLDER/PRINN/%j"
    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(slurm_partition=args.slurm_partition,
                               name="sort_hierarchy", slurm_time=200, mem_gb=10, slurm_additional_parameters={"account":"h2020deciderficarra"},slurm_array_parallelism=6)
    args = [args for c in candidates]
    executor.map_array(nested_patches, candidates, args)
    # for c in candidates:
    #     nested_patches(c, args[0])
    #nested_patches(13, args[13])

# def properties(candidate):
#     """
#     Retrieve properties of a candidate slide.

#     Args:
#         candidate (int): Index of the candidate slide.

#     Returns:
#         tuple: Tuple containing the properties of the candidate slide.
#             - real_name (str): Name of the candidate slide without the ".svs" extension.
#             - id (object): Identifier for the candidate slide.
#             - label (object): Label associated with the candidate slide.
#             - test (str): Type of test the slide belongs to.
#             - magnitude (object): Magnitude of the candidate slide.
#             - down (int): Downsample factor based on the microns per pixel (mpp) value.
#                             1 if mpp > 0.3, otherwise 0.
#     """
#     df = pd.read_csv("slide_properties.csv")
#     row = df.iloc[candidate]
#     real_name = row["0"].replace(".svs", "")
#     id = row["1"]
#     label = row["2"]
#     test = row["3"]
#     magnitude = row["4"]
#     mpp = row["5"]
#     # Determine the downsample factor based on mpp
#     if mpp > 0.3:
#         down = 1
#     else:
#         down = 0
#     return real_name, id, label, test, magnitude, down


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='sort_slide')
    parser.add_argument('--sourcex5', default="/mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard/x5/images/",
                        type=str, help='path to patches at 5x scale')
    parser.add_argument('--sourcex10', default="/mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard/x10/images/",
                        type=str, help='path to patches at 10x scale')
    parser.add_argument('--sourcex20', default="/mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard/x20/images/",
                        type=str, help='path to patches at 20x scale')
    parser.add_argument(
        '--slurm_partition', default="all_usr_prod", type=str, help='slurm partition')
    parser.add_argument('--step', default=10, type=int,
                        help='how many slides process within each job')
    parser.add_argument('--dest', default="/mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/step1_output/",
                        type=str, help='destination folder')
    args = parser.parse_args()
    return args



def nested_patches(candidate, args):
    """
    Process nested patches

    Args:
        candidate (int): Index of the candidate slide.
        args (argparse.Namespace): Parsed command line arguments.
    """
    dest = args.dest
    lista = glob.glob(os.path.join(args.sourcex20, "*"))
    print(len(lista))
    real_name = lista[candidate].split(os.sep)[-1]
    id = real_name
    test = ""
    down = 0
    label = ""

    levelx5pathdest = os.path.join(dest, test+id+"_"+str(label), "*.jpg")
    # levelx5path = args.sourcex5+real_name+"/"+str(3-down)+"/*"
    # levelx10path = args.sourcex10+real_name+"/"+str(2-down)+"/"
    # levelx20path = args.sourcex20+real_name+"/"+str(1-down)+"/"
    levelx5path = args.sourcex5+real_name+"/*"
    levelx10path = args.sourcex10+real_name+"/*"
    levelx20path = args.sourcex20+real_name+"/*"
    dest = os.path.join(dest, test+id+"_"+str(label))
    # if len(glob.glob(levelx5pathdest)) == len(glob.glob(levelx5path)):
    #     return
    # Process each patch at the x5 resolution
    x10processed=[]
    x20processed=[]
    totx20=len(glob.glob(levelx20path+"*.jpg"))
    x20list=glob.glob(levelx20path+"*.jpg")
    totx10=len(glob.glob(levelx10path+"*.jpg"))
    x10list=glob.glob(levelx10path+"*.jpg")
    for patch_x5_path in glob.glob(levelx5path):
        patch_name = os.path.basename(patch_x5_path).split(".")[0]
        os.makedirs(dest,exist_ok=True)
        shutil.copy(patch_x5_path, os.path.join(
            dest, os.path.basename(patch_x5_path)))
        newdestpath = os.path.join(dest, patch_name)
        print(newdestpath, flush=True)
        os.makedirs(newdestpath, exist_ok=True)
        # Generate the hierarchical structure and copy files accordingly
        x10processed, x20processed = hierarchy(patch_x5_path, levelx10path, levelx20path,
                  newdestpath, 2048/(1+down), 1,x10processed, x20processed)
        #x20processed.extend(x20processed_tmp)
    print('check differenza tra processed e rimanenti')
    x10remaining=[wsi for wsi in x10list if wsi.split('/')[-1] not in x10processed]

    #if len(x10remaining)>0:
    for patch in x10remaining:
        patch_name = os.path.basename(patch).split(".")[0]
        newdestpath = os.path.join(dest, patch_name)
        os.makedirs(newdestpath, exist_ok=True)
        shutil.copy(patch,os.path.join(newdestpath,os.path.basename(patch)))
        x10processed.append(patch_name)
        x10processed, x20processed = hierarchy(patch, levelx10path, levelx20path,
                os.path.join(newdestpath,newdestpath.split('/')[-1]), 2048/(1+down), 2,x10processed, x20processed)
            #x10remaining=[wsi for wsi in x10list if wsi.split('/')[-1] not in x10processed]
    
    x20remaining=[wsi for wsi in x20list if wsi.split('/')[-1] not in x20processed]    
    for patch in x20remaining:
        patch_name = os.path.basename(patch).split(".")[0]
        newdestpath = os.path.join(dest, patch_name,patch_name)
        print(newdestpath, flush=True)
        os.makedirs(newdestpath, exist_ok=True)
        shutil.copy(patch,os.path.join(newdestpath,os.path.basename(patch)))
        x20processed.append(os.path.basename(patch))
    print(len(x10processed),totx10)
    print(len(x20processed),totx20)    

args = get_args()
real_candidates = []
lista = glob.glob(os.path.join(args.sourcex20, "*"))
print(len(lista))
# Identify real candidates for slide processing
for candidate in range(len(lista)):
    dest = args.dest
    real_name = lista[candidate].split(os.sep)[-1]
    id = real_name
    test = ""
    down = 0
    label = ""

    levelx5pathdest = os.path.join(dest, test+id+str(label), ".jpg")
    #levelx20pathdest=os.path.join(dest, test+id+"_"+str(label), "/*/*.jpg")
    levelx20pathdest =os.path.join(dest, test+id+"_"+str(label), ".jpg")
    
    # levelx5path = args.sourcex5+real_name+"/"+str(3-down)+"/*"
    # levelx10path = args.sourcex10+real_name+"/"+str(2-down)+"/"
    # levelx20path = args.sourcex20+real_name+"/"+str(1-down)+"/"
    levelx5path = args.sourcex5+real_name+"/*"
    levelx10path = args.sourcex10+real_name+"/*"
    levelx20path = args.sourcex20+real_name+"/*"
    #print(levelx20path)
    dest = os.path.join(dest, test+id+"_"+str(label))
    print(id)
    if len(glob.glob(levelx20pathdest)) != len(glob.glob(levelx20path+'*.jpg')):
        real_candidates.append(candidate)
        # # print(levelx20pathdest,len(glob.glob(levelx20pathdest)))
        # print(levelx20path,len(glob.glob(levelx20path+'*.jpg')))
    else:
        print('already processed')
print(real_candidates)
prepareslide(real_candidates, args)
