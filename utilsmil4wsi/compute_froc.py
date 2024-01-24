## PyTorch
import torch
import os
# Torchvision
import random
import numpy as np
import wandb
import pandas as pd
from utils.parser import get_args
from utils.datasets import get_loaders
import submitit
from models.init import selectModel
import joblib
# Ensure that all operations are deterministic on GPU (if used) for reproducibility


def calculate_froc(model,testloader):
    model.eval()
    model=model.cuda()
    #FROC_data = np.zeros((4, len(testloader)), dtype=np.object)
    #FP_summary = np.zeros((2, len(testloader)), dtype=np.object)
    info=[]

    for caseNum,data in enumerate(testloader):
        data= data.cuda()
        x, edge_index,childof, batch_idx,level, edge_index_filtered,name,x_coord,y_coord,y = data.x, data.edge_index,data.childof, data.batch,data.level,data.edge_index_filtered,data.name,data.x_coord,data.y_coord,data.y
        #x=x[level==3]
        if name[0].split("_")[2]=="tumor":
            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2,edge_index3=data.edge_index_2,data.edge_index_3
            else:
                edge_index2=None
                edge_index3=None
            results = model(x, edge_index,level,childof,edge_index2,edge_index3)
            A= results["higher"][2].cpu().detach().numpy()

            x_coords=x_coord[level==3].cpu().detach().numpy()
            y_coords=y_coord[level==3].cpu().detach().numpy()
            scores=[]
            xc=[]
            yc=[]
            #for score,x,y in zip(A[:,0],x_coords,y_coords):
            #    for i in range(0,16):
            #        for j in range(0,16):
            #            scores.append(score)
            #            xc.append(x+(32*i))
            #            yc.append(y+(32*j))
            from sklearn.preprocessing import MinMaxScaler

            A=MinMaxScaler().fit_transform(A)
            data={"prob":A.reshape(-1),
                  "x":x_coords,
                  "y":y_coords,
                  "pred_y":torch.sigmoid( results["higher"][1][0][0]).cpu().detach().numpy(),
                  "slide":name}
            info.append(data)
    df=pd.DataFrame(info)
    return df


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def init_seed(seed):
    import torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True)
    eval('setattr(torch.backends.cudnn, "benchmark", True)')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def upload_file(file,run):
    print(file,run)

def processDataset(args):
    api = wandb.Api()
    filter_dict = {"tags": {"$in": ["froc"]}}
    runs=api.runs(path="gbont/estension",filters=filter_dict)
    for run in runs:
        modeltype=run._attrs["config"]["modeltype"]
        seed=run._attrs["config"]["seed"]
        runname= run._attrs["name"]
        path=run._attrs["config"]["datasetpath"]
        args.scale=run._attrs["config"]["scale"]

        args.datasetpath=path
        flag=0
        for file in run.files():
            if "attenzionifinalv2.joblib" in file.name:
                flag=1
                break
        if flag==1:
            print("already computed")
            continue
        print("not computed")

        args.modeltype=modeltype
        args.seed=seed
        init_seed(seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
        _,_,testloader=get_loaders(args)
        model=selectModel(args)
        run.file("model.pt").download(replace=True)


        model.load_state_dict(torch.load("model.pt"))
        try:

            with torch.no_grad():
                file=calculate_froc(model,testloader)
            #file.to_csv("attenzioni.csv")
            joblib.dump(file,"attenzionifinal.joblib")
            print(path)
            run.upload_file("attenzionifinal.joblib")
            #upload_file(file,runname)
        except Exception as e:
            print(e)

    wandb.finish()

def main():
    executor = submitit.AutoExecutor(folder="./loggraph", slurm_max_num_timeout=30)
    executor.update_parameters(
            mem_gb=10,
            gpus_per_node=1,
            tasks_per_node=1,  # one task per GPU
            slurm_cpus_per_gpu=1,
            nodes=1,
            exclude="aimagelab-srv-10",
            timeout_min=230,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition="prod",
            slurm_signal_delay_s=120,
            slurm_array_parallelism=10)
    executor.update_parameters(name="froc")
    args= get_args()

    #executor.submit(processDataset,args)
    processDataset(args)


if __name__ == '__main__':
    main()