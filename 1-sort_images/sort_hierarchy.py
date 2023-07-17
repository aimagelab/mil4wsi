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


def hierarchy(parent_path, x10path, x20path, dirpath, base_size, level):
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
        return
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
            if (not os.path.isfile(os.path.join(dirpath, newname))):
                shutil.copy(file, os.path.join(dirpath, newname))
            hierarchy(file, x10path, x20path, os.path.join(
                dirpath, newname.split(".")[0]), base_size, level+1)


def prepareslide(candidates, args):
    """
    Prepare slide data for processing

    Args:
        candidate (int): Index of the candidate slide.
        args (argparse.Namespace): Parsed command line arguments.
    """
    log_folder = "LOGFOLDER/%j"
    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(slurm_partition=args.slurm_partition,
                               name="sort_hierarchy", slurm_time=200, mem_gb=10, slurm_array_parallelism=3)
    args = [args for c in candidates]
    executor.map_array(nested_patches, candidates, args)


def properties(candidate):
    """
    Retrieve properties of a candidate slide.

    Args:
        candidate (int): Index of the candidate slide.

    Returns:
        tuple: Tuple containing the properties of the candidate slide.
            - real_name (str): Name of the candidate slide without the ".svs" extension.
            - id (object): Identifier for the candidate slide.
            - label (object): Label associated with the candidate slide.
            - test (str): Type of test the slide belongs to.
            - magnitude (object): Magnitude of the candidate slide.
            - down (int): Downsample factor based on the microns per pixel (mpp) value.
                            1 if mpp > 0.3, otherwise 0.
    """
    df = pd.read_csv("slide_properties.csv")
    row = df.iloc[candidate]
    real_name = row["0"].replace(".svs", "")
    id = row["1"]
    label = row["2"]
    test = row["3"]
    magnitude = row["4"]
    mpp = row["5"]
    # Determine the downsample factor based on mpp
    if mpp > 0.3:
        down = 1
    else:
        down = 0
    return real_name, id, label, test, magnitude, down


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='sort_slide')
    parser.add_argument('--sourcex5', default="SOURCEPATHx5",
                        type=str, help='path to patches at 5x scale')
    parser.add_argument('--sourcex10', default="SOURCEPATHx10",
                        type=str, help='path to patches at 10x scale')
    parser.add_argument('--sourcex20', default="SOURCEPATH20",
                        type=str, help='path to patches at 20x scale')
    parser.add_argument(
        '--slurm_partition', default="SLURM_PARTIITION", type=str, help='slurm partition')
    parser.add_argument('--step', default=10, type=int,
                        help='how many slides process within each job')
    parser.add_argument('--dest', default="DESTINATIONPATH",
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
    lista = glob.glob(os.path.join(args.sourcex5, "*"))
    real_name = lista[candidate].split(os.sep)[-1]
    id = real_name
    test = ""
    down = 0
    label = ""

    levelx5pathdest = os.path.join(dest, test+id+"_"+str(label), "*.jpg")
    levelx5path = args.sourcex5+real_name+"/"+str(3-down)+"/*"
    levelx10path = args.sourcex10+real_name+"/"+str(2-down)+"/"
    levelx20path = args.sourcex20+real_name+"/"+str(1-down)+"/"
    dest = os.path.join(dest, test+id+"_"+str(label))
    if len(glob.glob(levelx5pathdest)) == len(glob.glob(levelx5path)):
        return
    # Process each patch at the x5 resolution
    for patch_x5_path in glob.glob(levelx5path):
        patch_name = os.path.basename(patch_x5_path).split(".")[0]
        shutil.copy(patch_x5_path, os.path.join(
            dest, os.path.basename(patch_x5_path)))
        newdestpath = os.path.join(dest, patch_name)
        print(newdestpath, flush=True)
        os.makedirs(newdestpath, exist_ok=True)
        # Generate the hierarchical structure and copy files accordingly
        hierarchy(patch_x5_path, levelx10path, levelx20path,
                  newdestpath, 2048/(1+down), 1)


args = get_args()
real_candidates = []
lista = glob.glob(os.path.join(args.sourcex5, "*"))
# Identify real candidates for slide processing
for candidate in range(len(lista)):
    dest = args.dest
    real_name = lista[candidate].split(os.sep)[-1]
    id = real_name
    test = ""
    down = 0
    label = ""

    levelx5pathdest = os.path.join(dest, test+id+"_"+str(label), "*.jpg")
    levelx5path = args.sourcex5+real_name+"/"+str(3-down)+"/*"
    levelx10path = args.sourcex10+real_name+"/"+str(2-down)+"/"
    levelx20path = args.sourcex20+real_name+"/"+str(1-down)+"/"
    dest = os.path.join(dest, test+id+"_"+str(label))
    if len(glob.glob(levelx5pathdest)) != len(glob.glob(levelx5path)):
        real_candidates.append(candidate)

prepareslide(real_candidates, args)
