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

import argparse
import copy
import gc
import glob
import json
import os
from numpyencoder import NumpyEncoder

import numpy as np
import openslide
import pandas as pd
import geojson

def structure():
    """

    @return: json_skeleton
    """
    json_skeleton = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": 0
        },
        "properties": {
            "object_type": "annotation",
            "classification": {
                "name": "",
                "colorRGB": 0
            },
            "isLocked": False
        }
    }
    return json_skeleton

def thresholding(A, thresholds):
    n_thres = len(thresholds)
    thresholds = dict(sorted(thresholds.items(), key=lambda x:x[0]))

    max_A = A.max()
    intervals = []
    new_A = copy.copy(A)
    j=0
    for key, value in thresholds.items():
        if j == 0:
            right = 1.0*max_A
            left = value*max_A
            new_A[A>=left] = str(key)
            j+=1
        else:
            right = thresholds[key-1]*max_A
            left = value*max_A
            new_A[np.logical_and(A > left, A <= right)] = str(key)
            j+=1
        i = pd.Interval(left, right)
        intervals.append(i)

    right = value * max_A
    left = 0.0 * max_A
    new_A[A < right] = str(key)
    i = pd.Interval(left, right)
    intervals.append(i)

    idx = pd.IntervalIndex(intervals)
    intervals = pd.DataFrame({'intervals': idx, 'left': idx.left, 'right': idx.right})
    A = new_A
    return A, intervals

def compute_heatmap(A,x_array,y_array, res, bound):
    """
    @param:
        A: DataFrame of the attention map of single WSI
        coords: DataFrame with the coordinates information, columns=['x', 'y']
        res: string value that indicate the resolution of the coordinates
    @return:
    """
    encoder = {1: (2048, 2048),
               2: (1024, 1024),
               3: (512, 512)}

    thresholds = {1:0.8,
                  2:0.6,
                  3:0.4,
                  4:0.2,
                  5:0.1}

    A, intervals = thresholding(A, thresholds)

    patch_size = encoder[res.cpu().detach().numpy().item()]

    attention_json = []
    size=A.shape[0]
    for i in range(size):
        a_value = A[i].item()
        x = int(x_array[i])-int(bound[0])
        y = int(y_array[i])-int(bound[1])

        shift = 5

        top_left = [x+shift, y+shift]
        top_right = [x+patch_size[0]-shift, y+shift]
        bot_left = [x+shift, y+patch_size[1]-shift]
        bot_righ = [x+patch_size[0]-shift, y+patch_size[1]-shift]
        poly = [[top_left,
                top_right,
                bot_righ,
                bot_left,
                 top_left]]
        json_skeleton = structure()
        json_skeleton['geometry']['coordinates'] = poly
        json_skeleton['properties']['classification']['name'] = int(a_value)
        attention_json.append(json_skeleton)
    return attention_json

def processjson(A,x,y,name,levelmax):
    #A = pd.read_csv(args.attention_csv, index_col=0)

    root='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022/'
    destination_root="/mnt/beegfs/work/H2020DeciderFicarra/attentafrancy/json/"
    slide_path=root+name
    slide_path=slide_path[:-2]+".mrxs"

    json_path=destination_root+name
    json_path=json_path[:-2]+".json"
    #name = args.attention_csv.split('/')[10].split('.')[0]+'.json'

    #slide = args.attention_csv.split('/')[10].split('.')[0][:-2]+'.mrxs'
    wsi = openslide.open_slide(slide_path)
    x_bound = wsi.properties['openslide.bounds-x']
    y_bound = wsi.properties['openslide.bounds-y']
    json_heat = compute_heatmap(A,x,y, levelmax, bound=[x_bound, y_bound])

    os.makedirs(destination_root, exist_ok=True)
    with open(json_path, 'w') as f:
        geojson.dump(json_heat, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/pfi_45_180/x20s_4/56/')
    # parser.add_argument('--root', default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/pfi_45_180/x5s_3/29/')
    parser.add_argument('--annots', default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/annotations/x20s_4/56/')
    parser.add_argument('--slide_path', default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022/')
    parser.add_argument('--res', default='x20')
    parser.add_argument('--attention_csv', default=None)
    parser.add_argument('--parallel_job', default=False)

    args = parser.parse_args()
    return args




# coordinates = [[[]]]