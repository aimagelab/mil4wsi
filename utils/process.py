from utils.seed import init_seed
from utils.datasets import get_loaders
from utils.training import train
from utils.models import selectModelMulti

def processDataset(args):
    #init seeds
    init_seed(args)
    #prepare dataset and dataloaders
    train_loader,val_loader,test_loader=get_loaders(args)
    model= selectModelMulti(model_name=args.modeltype,num_layers=args.num_layers,c_hidden=args.c_hidden,temperature=args.temperature,beta=args.beta,lamb=args.lamb,layer_name=args.layer_name,c_out=args.n_classes,input_size=args.input_size,residual=args.residual,dropout=args.dropout)
    #start training
    bestmodel=train(model,trainloader=train_loader,valloader=val_loader,testloader=test_loader,epochs=args.n_epoch,args=args)
