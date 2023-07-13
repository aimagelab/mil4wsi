from utils.seed import init_seed
from utils.datasets import get_loaders
from utils.training import train
from models.init import selectModel


def processDataset(args):
    # Initialize seeds for reproducibility
    init_seed(args)
    # Prepare dataset and data loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    # Select the model based on the given arguments
    model = selectModel(args)
    # Start training
    bestmodel = train(model, trainloader=train_loader,
                      valloader=val_loader, testloader=test_loader, args=args)
