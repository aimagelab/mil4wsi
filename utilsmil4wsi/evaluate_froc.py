# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""
import openslide
import numpy as np
from scipy import ndimage as nd
from skimage import measure
import os
import sys
from tqdm import tqdm
import pandas as pd
import wandb
from skimage import exposure, io, img_as_ubyte, transform
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import cv2

def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[0]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0,0), 0, dims))
    distance = nd.distance_transform_edt(255 - pixelarray[:,:,0])
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2)
    return pixelarray[:,:,0]

def filtermask(slide, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    dims = slide.shape
    pixelarray = torch.Tensor(slide)
    pixelarray=pixelarray[:,:,0]
    pixelarray=pixelarray[None,:,:,None]
    pool=nn.modules.pooling.AvgPool2d((5,5),stride=1,padding=2)
    pixelarray=pool(pixelarray[:,:,:,0])*25
    pixelarray=pixelarray.squeeze().numpy()

    #pixelarray[pixelarray>100]=255
    #pixelarray[pixelarray<=100]=0
    #distance = nd.distance_transform_edt(pixelarray[:,:,0].astype(int))
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = pixelarray > 5
    #filled_image = nd.morphology.binary_fill_holes(binary).astype(int)
    #pixelarray[filled_image,:]=255
    #pixelarray[~filled_image,:]=0
    #evaluation_mask = measure.label(filled_image, connectivity = 2)
    mask=binary.astype(int)
    mask=mask[:,:,None]
    mask= np.concatenate([mask,mask,mask],axis=2)
    return mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file

    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image

    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(1,len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        if float(elems[1])>0.0001:
            Probs.append(float(elems[1]))
            Xcorr.append(int(elems[2]))
            Ycorr.append(int(elems[3]))
    return Probs, Xcorr, Ycorr


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            #HittedLabel = evaluation_mask[int(Ycorr[i]/32), int(Xcorr[i]/32)]
            #if mask[int(Ycorr[i]/pow(2,level)):(int(Ycorr[i]/pow(2,level))+32), int(Xcorr[i]/pow(2,level)):(int(Xcorr[i]/pow(2,level)+32))].max()>0:
            HittedLabel =evaluation_mask[int(Ycorr[i]/pow(2,level)):(int(Ycorr[i]/pow(2,level))+32), int(Xcorr[i]/pow(2,level)):(int(Xcorr[i]/pow(2,level)+32))].max()
            #HittedLabel=1
            if HittedLabel == 0:
                    print("fp",Probs[i],Xcorr[i], Ycorr[i],evaluation_mask.shape[0]*32,evaluation_mask.shape[1]*32)
                    FP_probs.append(Probs[i])
                    key = 'FP ' + str(FP_counter)
                    FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                    FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    print("tp ",HittedLabel,Probs[i],Xcorr[i], Ycorr[i],evaluation_mask.shape[0]*32,evaluation_mask.shape[1]*32)
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter+=1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells);
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    filter=[type(subitem)!=int for subitem in FROC_data[1]]
    newfrocdata1=FROC_data[1][filter]
    newfrocdata2=FROC_data[2][filter]

    unlisted_FPs = [item for sublist in newfrocdata1 for item in sublist]
    unlisted_TPs = [item for sublist in newfrocdata2 for item in sublist]

    total_FPs, total_TPs = [], []
    total_FPs.append(0)
    total_TPs.append(0)
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs),reverse=True)
    for Thresh in tqdm(all_probs[1:]):
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
        if (total_FPs[-1]/float(len(FROC_data[0])))>501:
            break

    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))
    return  total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity,name):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#444444')
    plt.show()
    plt.savefig(str(name)+"froc.png")


def saveevaluation(Probs,path):

    colors = [np.random.choice(range(256), size=3) for i in np.unique(Probs)]
    color_map = np.zeros((Probs.shape[0], Probs.shape[1],3))


    for region in np.unique(Probs):
        if region!=0:
            color_map[Probs==region]=colors[region-1]

    color_map = transform.resize(color_map, (color_map.shape[0], color_map.shape[1]), order=0)
    color_map=   exposure.rescale_intensity(color_map,out_range=(0,1))
    io.imsave(path, img_as_ubyte(color_map))


def saveattention(Probs,x_coords,y_coords,path,max_width,max_lenght):
    x_coords=np.array(x_coords)
    y_coords=np.array(y_coords)
    colors = [np.array([255,255,255]) for i in range(1)]
    colored_tiles = np.matmul(np.array(Probs)[:, None], colors[0][None, :])
    colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))
    color_map = np.zeros((int(max_lenght/32), int(max_width/32), 3))
    try:
        for i in range(x_coords.shape[0]):
            color_map[int(y_coords[i]/32),int( x_coords[i]/32)] = colored_tiles[i]
    except Exception as e:
        print(e)

    color_map = transform.resize(color_map, (color_map.shape[0], color_map.shape[1]), order=0)
    #io.imsave(path, img_as_ubyte(color_map))
    return color_map



def main():
    import glob
    mask_folder = MASK_FOLDER
    tot_cases= glob.glob( mask_folder)
    api = wandb.Api()
    filter_dict = {"tags": {"$in": ["froc"]}}

    runs=api.runs(path="gbont/estension",filters=filter_dict)
    index=0
    for run in runs:
        #if index==0:
        #    index=1
        #    continue
        try:
            run.file("attenzionifinal.joblib").download(replace=True)
        except:
            continue
        #wandb.init(resume="must",id=run.id,project="estension")

        df=joblib.load("attenzionifinal.joblib")
        tot_cases=len(df)

        EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
        L0_RESOLUTION = 0.243 # pixel resolution at level 0

        FROC_data = np.zeros((4, tot_cases-1-3), dtype=np.object)
        FP_summary = np.zeros((2, tot_cases-1-3), dtype=np.object)
        detection_summary = np.zeros((2, tot_cases-1-3), dtype=np.object)


        caseNum = 0
        for index,row in df.iterrows():
            if "tumor" not in row["slide"][0]:
                continue
            print(row["slide"][0])

            print("Evaluating Performance on image:", row["slide"][0][0:-4])
            sys.stdout.flush()

            try:

                number=row["slide"][0][5:-6]
                #if number !="055":
                #    continue
                #if os.path.exists(os.path.join("attentions", "tumor_"+number) + '_attentions.png'):
                #    continue
                Probs, Xcorr, Ycorr ,pred= row["prob"],row["x"],row["y"],row["pred_y"]
                if number== "117" or number== "030":
                    Probs=Probs[Ycorr > 15000]
                    Xcorr=Xcorr[Ycorr > 15000]
                    Ycorr=Ycorr[Ycorr > 15000]
                if pred<0.5:
                    Probs[:]=0
                Probs= MinMaxScaler().fit_transform(Probs.reshape(-1,1)).reshape(-1)

                #Probs2= np.copy(Probs)
                #Probs2[Probs>0.01]=1
                #Probs2[Probs<=0.01]=0

                filter= Probs>0
                Xcorr=Xcorr[filter]
                Ycorr=Ycorr[filter]
                Probs=Probs[filter]
                Xcorr=Xcorr/32
                Ycorr=Ycorr/32
                print("max:",Probs.max(),"min:",Probs.min())
                maskDIR = os.path.join(mask_folder, "tumor_"+number) + '_evaluation_mask.png'
                evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                #mask=saveattention(Probs2,Xcorr,Ycorr,"attentionslowfinal/"+str(number)+".png",max_width=evaluation_mask.shape[1],max_lenght=evaluation_mask.shape[0])
                #mask = filtermask(mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                #mask = cv2.resize(mask[:,:,0].astype(float), (evaluation_mask.shape[1],evaluation_mask.shape[0]), interpolation = cv2.INTER_AREA)

                Xcorr=Xcorr*32
                Ycorr=Ycorr*32

                ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                FROC_data[0][caseNum] = row["slide"][0]
                FP_summary[0][caseNum] = row["slide"][0]
                detection_summary[0][caseNum] = row["slide"][0]
                FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, True, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
                caseNum += 1

            except Exception as e:
                print(e)

        # Compute FROC curve
        total_FPs, total_sensitivity = computeFROC(FROC_data)
        data={"totalFP":total_FPs,"total_sensitivity":total_sensitivity}
        df=pd.DataFrame(data)
        # plot FROC curve
        plotFROC(total_FPs, total_sensitivity,"tot")
        data = [[x, y] for (x, y) in zip(total_FPs,total_sensitivity)]

        table = wandb.Table(data=data, columns = ["#FP", "sensitivity"])
        wandb.init(resume="must",id=run.id,project="estension")
        wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "#FP", "sensitivity", title="froc curve")})
        import time
        time.sleep(5)
        print("froc",froc)
        wandb.finish()
        #df.to_csv("fps.csv")



if __name__ == "__main__":
    log_folder = "log_test/%j"
    import submitit
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters( slurm_partition="prod",name="froc",slurm_time=300, cpus_per_task=10,mem_gb=20)
    #parameters= [args for source in sources]
    #process_slide(sources[0],dests[0],args)
    jobs = executor.submit(main)
    #main()