from utils.seed import init_seed
from utils.datasets import get_loaders
from utils.training import train
from models.init import selectModel

def processDataset(args):
    #init seeds
    init_seed(args)
    #prepare dataset and dataloaders
    train_loader,val_loader,test_loader=get_loaders(args)
    model= selectModel(args)
    #start training
    bestmodel=train(model,trainloader=train_loader,valloader=val_loader,testloader=test_loader,args=args)
