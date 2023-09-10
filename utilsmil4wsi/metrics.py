import numpy as np
import copy
from sklearn.metrics import roc_curve, roc_auc_score


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    """
    Computes ROC curve and AUC for each class in a multi-label classification task.

    Args:
        labels (numpy.ndarray): Ground truth labels.
        predictions (numpy.ndarray): Predicted probabilities.
        num_classes (int): Number of classes.
        pos_label (int): Positive label.

    Returns:
        aucs (list): AUC values.
        thresholds (list): Thresholds.
        thresholds_optimal (list): Optimal thresholds.
    """
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    labels = labels.reshape(-1, num_classes)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(
            fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    """
    Computes the optimal threshold based on the ROC curve coordinates and a loss function.

    Args:
        fpr (numpy.ndarray): False positive rates.
        tpr (numpy.ndarray): True positive rates.
        thresholds (numpy.ndarray): Threshold values.
        p (float): Weight parameter for the loss function.

    Returns:
        fpr[idx] (float): Optimal false positive rate.
        tpr[idx] (float): Optimal true positive rate.
        thresholds[idx] (float): Optimal threshold.
    """
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def computeMetrics(test_labels, test_predictions, num_classes=2, names=[]):
    """
    Computes evaluation metrics for multi-label classification.

    Args:
        test_labels (numpy.ndarray): Ground truth labels.
        test_predictions (numpy.ndarray): Predicted probabilities.
        num_classes (int): Number of classes.
        names (list): Class names.

    Returns:
        avg_score (float): Average score.
        auc_value[0] (float): AUC value.
        class_prediction_bag (numpy.ndarray): Class predictions.
    """
    if test_predictions.shape[0] == 0:
        return None, None, None
    auc_value, _, thresholds_optimal = multi_label_roc(
        test_labels, test_predictions, num_classes, pos_label=1)
    class_prediction_bag = copy.deepcopy(test_predictions)
    if num_classes > 1:
        for i in range(num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]
                                 >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]
                                 < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    else:
        probabilities = copy.deepcopy(test_predictions)
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
    test_labels = np.squeeze(test_labels)
    bag_score = 0
    for i in range(0, len(test_predictions)):
        bag_score = np.array_equal(
            test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_predictions)
    return avg_score, auc_value[0], class_prediction_bag
