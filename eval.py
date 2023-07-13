from utils.datasets import get_loaders
from models.init import selectModel
from utils.parser import get_args
from utils.experiments import *

args = get_args()
train_loader, val_loader, test_loader = get_loaders(args)
model = selectModel(args)
avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, _, _, _ = test(
    model, testloader=testloader)
print("acc: "+avg_score_higher_test+"auc: " + auc_value_higher_test)
