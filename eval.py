from utils.datasets import get_loaders
from models.init import selectModel
from utils.parser import get_args
from utils.experiments import *

# Get command-line arguments
args = get_args()

# Get data loaders
train_loader, val_loader, test_loader = get_loaders(args)

# Select model based on arguments
model = selectModel(args)

# Perform testing
avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, _, _, _ = test(
    model, testloader=test_loader)

# Print accuracy and AUC values
print("acc: "+avg_score_higher_test+"auc: " + auc_value_higher_test)
