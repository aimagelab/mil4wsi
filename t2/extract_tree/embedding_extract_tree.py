import sys
import argparse
import os

import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
import tqdm
import dino.vision_transformer as vits
import glob
import copy
from joblib import dump
from torchvision import models as torchvision_models
import dino.utils as utils




def getinfo(patch):
    """Extract x and y coordinates from patch path

    Args:
        patch (str): Path of the patch.

    Returns:
        tuple: Tuple containing the x and y coordinates extracted from the patch path.
    """
    infos = patch.split(os.sep)[-1].split("_")
    y = int(infos[-1].split(".")[0])
    x = int(infos[-3])
    return x, y


def encapsulate_patch_info(parent_id, x, y, id, shift, level, embedding, path):
    """
    Encapsulate patch information into a dictionary

    Args:
        parent_id (str): The ID of the parent patch.
        x (int): The x-coordinate of the patch.
        y (int): The y-coordinate of the patch.
        id (str): The ID of the patch.
        shift (int): The shift value of the patch.
        level (int): The level of the patch.
        embedding (np.ndarray): The embedding of the patch.
        path (str): The path of the patch.

    Returns:
        kinfo (dict): A dictionary containing the patch information.
    """
    kinfo = {}
    kinfo["childof"] = parent_id
    kinfo["x"] = x
    kinfo["id"] = id
    kinfo["y"] = y
    kinfo["shift"] = shift
    kinfo["level"] = level
    kinfo["embedding"] = embedding
    kinfo["path"] = path
    return kinfo


def getembedding(models, img, level):
    """
    Get the embedding for an image patch from the specified resolution level of the models

    Args:
        models (List[Model]): The models used for embedding.
        img (Image.Image): The image patch for which embedding is computed.
        level (int): The resolution level from which embedding is extracted.

    Returns:
        embedding (np.ndarray): The embedding vector for the image patch.
    """
    #print(img)
    level = 3-level
    #img = Image.open(img)
    img = VF.to_tensor(img).float().cuda()
    img = img.view(1, 3, 256, 256)
    embedding = models[level](img).detach().cpu().numpy()
    return embedding


def properties(candidate, path):
    """
    Get properties of a candidate from a CSV file

Args:
        candidate (int): The index of the candidate in the CSV file.
        path (str): The path to the CSV file.

    Returns:
        real_name (str): The name of the image.
        candidate (int): The index of the candidate.
        label (str): The label of the candidate.
        test (str): The phase of the candidate.
        down (int): The down value.
    """
    df = pd.read_csv(path)
    row = df.iloc[candidate]
    real_name = row["image"]
    label = row["label"]
    test = row["phase"]
    down = 0
    return real_name, id, label, test, down

def get_children(parent_id, x_base, y_base, basepath, allowedlevels, level, models, kinfos, infosConcat, base_shift):
    """
    Recursively get children patches and their embeddings

    Args:
        parent_id (str): The ID of the parent patch.
        x_base (int): The base x-coordinate.
        y_base (int): The base y-coordinate.
        basepath (str): The base path to the image patches.
        allowedlevels (List[int]): The allowed resolution levels.
        level (int): The current resolution level.
        models (List[Model]): The models used for embedding.
        kinfos (List[Dict[str, Any]]): The list of patch information dictionaries.
        infosConcat (List[Any]): The list to store concatenated embeddings.
        base_shift (int): The base shift value.

    Returns:
        kinfos (List[Dict[str, Any]]): The list of patch information dictionaries.
        infosConcat (List[Any]): The list of concatenated embeddings.
    """

    # If no children return list unchanged
    if level == 4:
        return kinfos, infosConcat
    # Calculate the shift value for the current level
    shift = int(base_shift/2**level)
    # Calculate the updated coordinates based on the shift value
    upperleft = (x_base, y_base)
    lowleft = (x_base, y_base+shift)
    upperright = (x_base+shift, y_base)
    lowright = (x_base+shift, y_base+shift)
    if parent_id is not None:
        parent = kinfos[parent_id]
    # Iterate over patches in the area enclosed by these coordinates
    for patch in [upperleft, lowleft, upperright, lowright]:
        x, y = patch
        path = glob.glob(os.path.join(
            basepath, "*_x_"+str(x)+"_y_"+str(y)+".jpg"))
        # If file is not present continue
        if len(path) == 0:
            continue
        path = path[0]
        image = Image.open(path)
        if level in allowedlevels:
            x, y = getinfo(path)
            embedding = getembedding(models, image, level)
            if parent_id is not None:
                concatened = np.concatenate(
                    [parent["embedding"], embedding], axis=-1)
                infosConcat.extend(concatened)
            kinfo = encapsulate_patch_info(parent_id=parent_id, x=x, y=y, id=len(
                kinfos), shift=shift, level=level, embedding=embedding, path=path)
            kinfos.append(kinfo)
            kinfos, infosConcat = get_children(kinfo["id"], x, y, path.split(
                ".")[0], allowedlevels, level+1, models, kinfos, infosConcat, base_shift)
        else:
            kinfos, infosConcat = get_children(parent_id, x, y, path.split(
                ".")[0], allowedlevels, level+1, models, kinfos, infosConcat, base_shift)

    return kinfos, infosConcat

  #


def search_neighboors(infos):
    """
    Search for neighboring patches and add the information to the DataFrame

    Args:
        infos (List[Dict]): List of dictionaries per patch to search.

    Returns:
        df(pd.DataFrame): The updated DataFrame with neighboring patches information.
    """
    df = pd.DataFrame(infos)
    df["nearsto"] = None
    print("scanning for neighboors")
    for level in tqdm.tqdm(range(0, 4)):
        df2 = df[df["level"] == level]
        for idx in range(df2.shape[0]):
            patch = df2.iloc[idx]
            shift = patch["shift"]
            x = patch["x"]
            y = patch["y"]
            lista = []
            df3 = df2.query("(x=="+str(x+shift)+" & y=="+str(y)+") | (x=="+str(x+shift)+" & y=="+str(y + shift)+") | (x=="+str(x)+" & y=="+str(y+shift)+")  | (x=="+str(x-shift)+" & y=="+str(
                y+shift)+")  | (x=="+str(x-shift)+" & y=="+str(y)+")  | (x=="+str(x-shift)+" & y=="+str(y-shift)+")  | (x=="+str(x)+" & y=="+str(y-shift)+")  | (x=="+str(x+shift)+" & y=="+str(y-shift)+")")
            for idx2 in range(df3.shape[0]):
                neighboor = df3.iloc[idx2]
                lista.append(neighboor["id"])
            df.at[df.index[df["id"] == patch["id"]][0], "nearsto"] = lista
    return df


def create_matrix(df):
    """
    Create an adjacency matrix based on the neighboring patches

    Args:
        df (pd.DataFrame): The DataFrame containing patch information.

    Returns:
        matrix (torch.Tensor): The adjacency matrix.
    """
    matrix = torch.zeros(size=[df.shape[0], df.shape[0]])
    for idx in range(df.shape[0]):
        patch = df.iloc[idx]
        i = patch["id"]
        neighbors = patch["nearsto"]
        parent = patch["childof"]
        for j in neighbors:
            matrix[int(i), int(j)] = 1
            matrix[int(j), int(i)] = 1
        if parent is not None:
            if not np.isnan(parent):
                matrix[int(i), int(parent)] = 1
                matrix[int(parent), int(i)] = 1
    return matrix


def compute_tree_feats_Slide(real_name, label, test, args, models, save_path=None, base_shift=2048):
    """Compute tree features for a slide

    Args:
        real_name (str): The name of the slide.
        label (str): The label of the candidate.
        test (str): The phase of the candidate.
        args (Any): The arguments for computation.
        models (List[Model]): The models used for computation.
        save_path (str, optional): The path to save the computed features. Defaults to None.
        base_shift (int, optional): The base shift value. Defaults to 2048.
    """

    allowedlevels = args.levels
    level = 1
    shift = int(base_shift/2**level)
    with torch.no_grad():
        # Initialize lists
        torch.backends.cudnn.enabled = False
        infos = []
        infos_concat = []
        dest = os.path.join(save_path, test, real_name+"_"+str(label))
        low_patches = glob.glob(os.path.join(
            args.extractedpatchespath, real_name, '*.jpg'))
        for path in tqdm.tqdm(low_patches):
            # Extract info about the patch
            x, y = getinfo(path)
            if level in allowedlevels:
                embedding = getembedding(models, path, level)
                kinfo = encapsulate_patch_info(parent_id=None, path=path, x=x, y=y, id=len(
                    infos), shift=shift, level=level, embedding=embedding)
                infos.append(kinfo)
                infos, infos_concat = get_children(kinfo["id"], x, y, path.split(
                    ".")[0], args.levels, level+1, models, infos, infos_concat, base_shift)
            else:
                infos, infos_concat = get_children(None, x, y, path.split(
                    ".")[0], args.levels, level+1, models, infos, infos_concat, base_shift)
        # Infos should contain a list of dictionaries per patch
        infos = search_neighboors(infos)
        matrix = create_matrix(infos)
        os.makedirs(dest, exist_ok=True)

        dump(infos, os.path.join(dest, "embeddings.joblib"))
        torch.save(matrix, os.path.join(dest, "adj.th"))
        del infos


def load_parameters(model, path, name, device):
    """
    Load parameters for a model

    Args:
        model (Any): The model to load parameters into.
        path (str): The path to the parameter file.
        name (str): The name of the model.
        device (torch.device): The device to load the parameters on.

    Returns:
        model (): The model with loaded parameters.
    """
    state_dict_weights = torch.load(path, map_location=device)
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = model.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


def main():
    """Main function for computing features from DINO embedder"""
    parser = argparse.ArgumentParser(
        description='Compute features from DINO embedder')
    # Parse command-line arguments
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of threads for datalodaer')
    parser.add_argument('--norm_layer', default='instance',
                        type=str, help='Normalization layer [instance]')
    parser.add_argument("--extractedpatchespath",
                        default="HIERARCHICALSOURCEPATH", type=str)
    parser.add_argument("--savepath", type=str, default="DESTINATIONPATH")
    parser.add_argument("--job_number", type=int, default=-1)
    parser.add_argument('--arch', default='vit_small',
                        type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights1', default='CHECKPOINTDINO20x',
                        type=str, help="embedder trained at level 1 (scale x20).")
    parser.add_argument('--pretrained_weights2', default='CHECKPOINTDINO10x',
                        type=str, help="embedder trained at level 2 (scale x10)")
    parser.add_argument('--pretrained_weights3', default='CHECKPOINTDINO5x',
                        type=str, help="embedder trained at level 3 (scale x5).")
    parser.add_argument('--n_last_blocks', default=4, type=int,
                        help="""Concatenate [CLS] tokens for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    args = parser.parse_args()
    model = buildnetwork(args)

  # Build the network model
    models = []
    weights = [args.pretrained_weights1,
               args.pretrained_weights2, args.pretrained_weights3]
    for idx in range(3):
        # Create a deep copy of the model for each level
        net = copy.deepcopy(model)
        net = net.cuda()
        net.eval()
        # Load pre-trained weights into the model
        utils.load_pretrained_weights(
            net, weights[idx], args.checkpoint_key, args.arch, args.patch_size)
        models.append(net)

    print(f"Model {args.arch} built.")
    print('Use pretrained features.')
    bags_path = os.path.join(args.extractedpatchespath, "*")
    feats_path = args.savepath
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    num_bags = len(bags_list)
    # Process features for each slide
    for slideNumber in range(num_bags):
        compute_tree_feats_Slide(
            slideNumber, args, bags_list, models, feats_path)


def processSlide(start, args):
    """
    Process slides starting from the specified index

    Args:
        start (int): The starting index of the slides.
        args (Any): The command-line arguments.
    """
    # Build the network model
    model = buildnetwork(args)
    models = []
    weights = [args.pretrained_weights1,
               args.pretrained_weights2, args.pretrained_weights3]
    for idx in range(3):
        # Create a deep copy of the model for each level
        net = copy.deepcopy(model)
        net = net.cuda()
        net.eval()
        if args.model == "dino":
            # Load pre-trained weights into the model
            utils.load_pretrained_weights(
                net, weights[idx], args.checkpoint_key, args.arch, args.patch_size)
        models.append(net)
    print('Use pretrained features.')
    # bags_path = os.path.join(args.extractedpatchespath,"*")
    feats_path = args.savepath
    os.makedirs(feats_path, exist_ok=True)
    # bags_list = glob.glob(bags_path)
    for slideNumber in range(start, start+args.step):
        real_name, id, label, test, down = properties(
            slideNumber, args.propertiescsv)
        print(real_name, label, test, down,feats_path)
        if os.path.isfile(os.path.join(feats_path, test, real_name+"_"+str(label), "embeddings.joblib")):
            print("skip")
            continue
        else:
            compute_tree_feats_Slide(
                real_name, label, test, args, models, feats_path, 4096/(1+down))


def buildnetwork(args):
    """
    Build the network model based on the provided arguments.

    Args:
        args (Any): The command-line arguments.

    Returns:
       model (Any): The built network model.
    """
    # If the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.model == "dino":
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](
                patch_size=args.patch_size, num_classes=0)
            embed_dim = model.embed_dim * \
                (args.n_last_blocks + int(args.avgpool_patchtokens))
        # If the network is a XCiT
        elif "xcit" in args.arch:
            model = torch.hub.load(
                'facebookresearch/xcit:main', args.arch, num_classes=0)
            embed_dim = model.embed_dim
        # Otherwise, check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch]()
            embed_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)
        return model


if __name__ == '__main__':
    main()
