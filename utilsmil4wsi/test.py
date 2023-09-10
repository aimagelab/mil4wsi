import torch
import numpy as np
from utilsmil4wsi.metrics import computeMetrics


def test(model, testloader):
    """
    Perform testing on the model.

    Args:
        model (torch.nn.Module): Model to be tested.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        avg_score_higher (float): Average score for higher-level predictions.
        avg_score_lower (float): Average score for lower-level predictions.
        auc_value_higher (float): AUC value for higher-level predictions.
        auc_value_lower (float): AUC value for lower-level predictions.
        class_prediction_bag_higher (numpy.ndarray): Class predictions for higher-level predictions.
        class_prediction_bag_lower (numpy.ndarray): Class predictions for lower-level predictions.
        test_labels (numpy.ndarray): Ground truth labels.
    """
    model.eval()
    results = []
    test_predictions0 = []
    test_predictions1 = []
    test_labels = []
    names = []
    # Iterate over the test data
    for _, data in enumerate(testloader):
        data = data.cuda()
        x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level

        # Check if additional edge indices are present
        if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
            edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
        else:
            edge_index2 = None
            edge_index3 = None
        # Forward pass through the model
        results = model(x, edge_index, level, childof,
                        edge_index2, edge_index3)
        bag_label = data.y.float().squeeze().cpu().numpy()

        # Convert bag label to numpy array
        if model.classes == 2:
            if bag_label == 1:
                bag_label = torch.LongTensor(
                    [[0, 1]]).float().squeeze().cpu().numpy()
            else:
                bag_label = torch.LongTensor(
                    [[1, 0]]).float().squeeze().cpu().numpy()

        # Append the test labels and predictions
        test_labels.extend([bag_label])
        preds = model.predict(results)
        test_predictions0.extend([(preds[0]).squeeze().cpu().detach().numpy()])

        if preds[1] is not None:
            test_predictions1.extend(
                [(preds[1]).squeeze().cpu().detach().numpy()])

    # Convert the test labels and predictions to numpy arrays
    test_labels = np.array(test_labels)
    test_predictions0 = np.array(test_predictions0)
    test_predictions1 = np.array(test_predictions1)

    # Compute metrics for higher-level predictions
    avg_score_higher, auc_value_higher, class_prediction_bag_higher = computeMetrics(
        test_labels, test_predictions0, model.classes, names)

    # Compute metrics for lower-level predictions if available
    if test_predictions1.shape[0] != 0:
        avg_score_lower, auc_value_lower, class_prediction_bag_lower = computeMetrics(
            test_labels, test_predictions1, model.classes, names)
    else:
        avg_score_lower = 0
        auc_value_lower = 0
        class_prediction_bag_lower = 0

    # Set the model back to training mode
    model.train()

    # Return the computed metrics and predictions
    return avg_score_higher, avg_score_lower, auc_value_higher, auc_value_lower, class_prediction_bag_higher, class_prediction_bag_lower, test_labels
