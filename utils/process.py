from utils.seed import init_seed
from utils.datasets import get_loaders
from utils.training import train
from models.init import selectModel
from utils.test import test
import torch


def processDataset(args):
    # Initialize seeds for reproducibility
    init_seed(args)
    #prepare dataset and dataloaders
    train_loader,val_loader,test_loader=get_loaders(args)
    model= selectModel(args)
    #start training
    bestmodel=train(model,trainloader=train_loader,valloader=val_loader,testloader=test_loader,args=args)

def eval(args):
    #init seeds
    init_seed(args)
    model= selectModel(args)
    state_dict_weights = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(state_dict_weights, strict=False)
    except:
        print("error loading")
    model.eval()
    #prepare dataset and dataloaders
    train_loader,val_loader,test_loader=get_loaders(args)
    #start training
    avg_score_higher_test,avg_score_lower_test,auc_value_higher_test,auc_value_lower_test,predictions,_,labels=test(model,testloader=test_loader)
    print("ACCURACY:"+str(avg_score_higher_test))
    print("AUC:"+str(auc_value_higher_test))
