import torch
import numpy as np
from utils.metrics import computeMetrics


def test(model, testloader):
    model.eval()
    results = []
    test_predictions0 = []
    test_predictions1 = []
    test_labels = []
    names = []
    for _, data in enumerate(testloader):
        data = data.cuda()
        x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level
        if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
            edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
        else:
            edge_index2 = None
            edge_index3 = None

        results = model(x, edge_index, level, childof,
                        edge_index2, edge_index3)
        bag_label = data.y.float().squeeze().cpu().numpy()
        if model.classes == 2:
            if bag_label == 1:
                bag_label = torch.LongTensor(
                    [[0, 1]]).float().squeeze().cpu().numpy()
            else:
                bag_label = torch.LongTensor(
                    [[1, 0]]).float().squeeze().cpu().numpy()
        test_labels.extend([bag_label])
        preds = model.predict(results)
        test_predictions0.extend([(preds[0]).squeeze().cpu().detach().numpy()])

        if preds[1] is not None:
            test_predictions1.extend(
                [(preds[1]).squeeze().cpu().detach().numpy()])
    test_labels = np.array(test_labels)
    test_predictions0 = np.array(test_predictions0)
    test_predictions1 = np.array(test_predictions1)
    avg_score_higher, auc_value_higher, class_prediction_bag_higher = computeMetrics(
        test_labels, test_predictions0, model.classes, names)
    if test_predictions1.shape[0] != 0:
        avg_score_lower, auc_value_lower, class_prediction_bag_lower = computeMetrics(
            test_labels, test_predictions1, model.classes, names)
    else:
        avg_score_lower = 0
        auc_value_lower = 0
        class_prediction_bag_lower = 0
    model.train()
    return avg_score_higher, avg_score_lower, auc_value_higher, auc_value_lower, class_prediction_bag_higher, class_prediction_bag_lower, test_labels
