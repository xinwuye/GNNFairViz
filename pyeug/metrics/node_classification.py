from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np


def micro_f1_std_dev(prediction, labels, groups):
    """
    Calculate the standard deviation of Micro-F1 scores for each sensitive subgroup.
    
    Parameters:
    prediction (np.ndarray): A 1D array of predicted labels from the machine learning model.
    labels (np.ndarray): A 1D array of ground truth labels.
    groups (np.ndarray): A 1D array containing categorical values where each category corresponds to a sensitive subgroup.

    Returns:
    float: The standard deviation of Micro-F1 scores across the unique groups.
    
    Note:
    The input arrays must be of the same length, with each element corresponding across the arrays.
    """
    unique_groups = np.unique(groups)
    micro_f1_scores = []

    for group in unique_groups:
        group_indices = np.where(groups == group)
        group_pred = prediction[group_indices]
        group_labels = labels[group_indices]

        micro_f1 = f1_score(group_labels, group_pred, average='micro')
        micro_f1_scores.append(micro_f1)

    std_dev = np.std(micro_f1_scores)
    return std_dev


# def surrogate_di(prediction, labels, groups):
#     """
#     Calculate the mean of the standard deviations of the proportion of positive predictions 
#     for each sensitive subgroup within each label. This is a multiclass classification fairness metric.

#     Parameters:
#     prediction (np.ndarray): A 1D numpy array of predicted labels from the machine learning model.
#     labels (np.ndarray): A 1D numpy array of true labels.
#     groups (np.ndarray): A 1D numpy array containing categorical group membership for each instance.

#     Returns:
#     float: The mean of the standard deviations of positive prediction proportions across labels.

#     Notes:
#     - This metric is a multiclass generalization of surrogate of the Disparate Impact (DI) metric.
#     - The input arrays must be of the same length and correspond to each other element-wise.
#     - It is assumed that the labels array contains the correct class for each instance, and
#       the groups array contains the sensitive attribute by which fairness is being assessed.
#     """
#     unique_groups = np.unique(groups)
#     unique_labels = np.unique(labels)
    
#     std_devs = []

#     for label in unique_labels:
#         positive_proportions = []

#         for group in unique_groups:
#             group_indices = np.where(groups == group)
#             group_predictions = prediction[group_indices]

#             # Proportion of positive predictions for the current group and label
#             positive_proportion = np.mean(group_predictions == label)
#             positive_proportions.append(positive_proportion)

#         # Standard deviation of proportions for the current label
#         std_dev = np.std(positive_proportions)
#         std_devs.append(std_dev)

#     # Mean of standard deviations across all labels
#     mean_std_dev = np.mean(std_devs)
#     return mean_std_dev


def efpr(prediction, labels, groups):
    """
    Calculate the mean of the standard deviations of the False Positive Rate (EFPR) 
    across all labels for multiclass classification tasks.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The mean standard deviation of FPR across all labels.

    Notes:
    - EFPR is calculated for each label and group as the ratio of false positives to total non-positives.
    - The function computes the standard deviation of FPR across groups for each label and then 
      takes the mean of these standard deviations.
    """
    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)
    
    efpr_std_devs = []

    for label in unique_labels:
        fpr_values = []
        
        for group in unique_groups:
            # Define true negatives and false positives for the group and label
            tn = np.sum((prediction[groups == group] != label) & (labels[groups == group] != label))
            fp = np.sum((prediction[groups == group] == label) & (labels[groups == group] != label))
            
            # Calculate FPR for the group
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_values.append(fpr)

        # Compute the standard deviation of FPR for the current label
        std_dev = np.std(fpr_values)
        efpr_std_devs.append(std_dev)

    # Calculate the mean of the standard deviations across all labels
    mean_efpr_std_dev = np.mean(efpr_std_devs)
    return mean_efpr_std_dev


def efnr(prediction, labels, groups):
    """
    Calculate the mean of the standard deviations of the False Negative Rate (EFNR) 
    across all labels for multiclass classification tasks.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The mean standard deviation of FNR across all labels.

    Notes:
    - EFNR is calculated for each label and group as the ratio of false negatives to total actual positives.
    - The function computes the standard deviation of FNR across groups for each label and then 
      takes the mean of these standard deviations.
    """
    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)
    
    efnr_std_devs = []

    for label in unique_labels:
        fnr_values = []
        
        for group in unique_groups:
            # Define false negatives and true positives for the group and label
            fn = np.sum((prediction[groups == group] != label) & (labels[groups == group] == label))
            tp = np.sum((prediction[groups == group] == label) & (labels[groups == group] == label))
            
            # Calculate FNR for the group
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fnr_values.append(fnr)

        # Compute the standard deviation of FNR for the current label
        std_dev = np.std(fnr_values)
        efnr_std_devs.append(std_dev)

    # Calculate the mean of the standard deviations across all labels
    mean_efnr_std_dev = np.mean(efnr_std_devs)
    return mean_efnr_std_dev


# def etpr(prediction, labels, groups):
#     """
#     Calculate the mean of the standard deviations of the True Positive Rate (ETPR) 
#     across all labels for multiclass classification tasks.

#     Parameters:
#     predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
#     labels (np.ndarray): A 1D numpy array of the ground truth labels.
#     groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

#     Returns:
#     float: The mean standard deviation of ETPR across all labels.

#     Notes:
#     - ETPR is calculated for each label and group as the ratio of true positives to total actual positives.
#     - The function computes the standard deviation of ETPR across groups for each label and then 
#       takes the mean of these standard deviations.
#     """
#     unique_groups = np.unique(groups)
#     unique_labels = np.unique(labels)
    
#     etpr_std_devs = []

#     for label in unique_labels:
#         tpr_values = []
        
#         for group in unique_groups:
#             # Define true positives and false negatives for the group and label
#             tp = np.sum((prediction[groups == group] == label) & (labels[groups == group] == label))
#             fn = np.sum((prediction[groups == group] != label) & (labels[groups == group] == label))
            
#             # Calculate TPR for the group
#             tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#             tpr_values.append(tpr)

#         # Compute the standard deviation of TPR for the current label
#         std_dev = np.std(tpr_values)
#         etpr_std_devs.append(std_dev)

#     # Calculate the mean of the standard deviations across all labels
#     mean_etpr_std_dev = np.mean(etpr_std_devs)
#     return mean_etpr_std_dev


def delta_dp_metric(predictions, labels, groups):
    """
    Calculate the ΔDP (Delta Disparate Impact) metric for multiclass classification, 
    which is the average standard deviation of the positive prediction probabilities across 
    sensitive groups for each class.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The ΔDP metric, representing the mean of the standard deviations of the positive 
           prediction probabilities across groups for each class.

    Notes:
    - The metric reflects fairness in terms of the variability of the model's predictions across different groups.
    - Lower values indicate more fairness as they reflect less variability in positive prediction rates among groups.
    """
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    label_group_probs = []

    # Compute the probability P(Y_hat = y | S = s) for each label and group
    for label in unique_labels:
        group_probs = []
        for group in unique_groups:
            # Calculate the number of occurrences where the prediction is equal to the current label within the current group
            group_predictions = predictions[groups == group]
            prob = np.mean(group_predictions == label)
            group_probs.append(prob)
        
        # Calculate the standard deviation of the probabilities for the current label across groups
        std_dev = np.std(group_probs)
        label_group_probs.append(std_dev)
    
    # Calculate the mean of the standard deviations across all labels
    delta_dp = np.mean(label_group_probs)
    return delta_dp


def max_diff_delta_dp(predictions, labels, groups):
    """
    Calculate the multiclass fairness metric that measures the maximum difference in 
    positive prediction probabilities between each pair of sensitive groups for each label.
    The final metric is the mean of these maximum differences across all labels.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The mean of the maximum differences in prediction probabilities across all pairs of sensitive groups for each label.

    Notes:
    - This metric assesses the maximum disparity in prediction probabilities across different groups.
    - Lower values indicate more fairness, suggesting smaller disparities in prediction probabilities among groups.
    """
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    max_diffs = []

    # Compute the maximum difference in probabilities P(Y_hat = y | S = s) for each label
    for label in unique_labels:
        group_probs = []

        # Calculate the probabilities for the current label within each group
        for group in unique_groups:
            group_predictions = predictions[groups == group]
            prob = np.mean(group_predictions == label)
            group_probs.append(prob)
        
        # Calculate the maximum difference in probabilities for the current label across all group pairs
        max_diff = max([abs(p1 - p2) for p1 in group_probs for p2 in group_probs])
        max_diffs.append(max_diff)
    
    # Calculate the mean of the maximum differences across all labels
    delta_max_diff = np.mean(max_diffs)
    return delta_max_diff


def delta_eo_metric(predictions, labels, groups):
    """
    Calculate the ΔEO (Delta Equalized Odds) metric for multiclass classification. 
    This metric averages the standard deviation of the conditional probabilities 
    P(Ŷ = y | Y = y, S = s) for each label y across all sensitive subgroups s.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The ΔEO metric, representing the mean of the standard deviations of the 
           conditional probabilities across sensitive groups for each label.

    Notes:
    - ΔEO measures fairness in terms of equalized opportunity across sensitive groups.
    - It reflects how consistently the model predicts each class across different groups.
    - Lower values indicate more fairness, suggesting similar true positive rates across groups.
    """
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    eo_std_devs = []

    for label in unique_labels:
        conditional_probs = []
        
        for group in unique_groups:
            # Indices where the true label is the current label for the current group
            true_label_indices = (labels == label) & (groups == group)
            true_label_group_count = np.sum(true_label_indices)
            
            # Avoid division by zero
            if true_label_group_count == 0:
                continue

            # Calculate the conditional probability P(Y_hat=y | Y=y, S=s) for the current group
            correct_pred_for_group = np.sum((predictions[true_label_indices] == label))
            conditional_prob = correct_pred_for_group / true_label_group_count
            conditional_probs.append(conditional_prob)

        # Calculate the standard deviation of the conditional probabilities for the current label
        std_dev = np.std(conditional_probs) if len(conditional_probs) > 1 else 0
        eo_std_devs.append(std_dev)

    # Calculate the mean of the standard deviations across all labels
    mean_eo_std_dev = np.mean(eo_std_devs)
    return mean_eo_std_dev


def delta_accuracy_metric(predictions, labels, groups):
    """
    Calculate the Delta Accuracy (Δacc) metric, which measures the maximum disparity 
    in accuracy between any two sensitive groups.

    Parameters:
    predictions (np.ndarray): A 1D array of predictions from the model.
    labels (np.ndarray): A 1D array of the true labels.
    groups (np.ndarray): A 1D array indicating the group for each instance.

    Returns:
    float: The calculated Δacc metric value.
    """
    unique_groups = np.unique(groups)
    accuracies = []

    for group in unique_groups:
        # Select the predictions and labels for the current group
        group_indices = np.where(groups == group)
        group_predictions = predictions[group_indices]
        group_labels = labels[group_indices]

        # Calculate the accuracy for the current group
        group_accuracy = accuracy_score(group_labels, group_predictions)
        accuracies.append(group_accuracy)

    # Calculate the difference between the maximum and minimum accuracies
    delta_acc = max(accuracies) - min(accuracies)
    return delta_acc


def sigma_accuracy_metric(predictions, labels, groups):
    """
    Calculate the standard deviation of accuracy (σacc) across all sensitive groups.

    Parameters:
    predictions (np.ndarray): A 1D array of predictions from the model.
    labels (np.ndarray): A 1D array of the true labels.
    groups (np.ndarray): A 1D array indicating the group for each instance.

    Returns:
    float: The standard deviation of the group accuracies.
    """
    unique_groups = np.unique(groups)
    accuracies = []

    for group in unique_groups:
        # Select the predictions and labels for the current group
        group_indices = np.where(groups == group)
        group_predictions = predictions[group_indices]
        group_labels = labels[group_indices]

        # Calculate the accuracy for the current group
        group_accuracy = accuracy_score(group_labels, group_predictions)
        accuracies.append(group_accuracy)

    # Compute the mean accuracy
    mean_accuracy = np.mean(accuracies)
    
    # Compute the standard deviation of accuracies
    sigma_acc = np.sqrt(np.mean((accuracies - mean_accuracy) ** 2))
    return sigma_acc


# def delta_ted_metric(predictions, labels, groups):
#     """
#     Calculate the multiclass extension of the treatment equality (ΔTED) metric. 
#     This function computes the mean of the standard deviations of the ratios of false 
#     positives to false negatives for each label within each group.

#     Parameters:
#     predictions (np.ndarray): A 1D array of predictions from the model.
#     labels (np.ndarray): A 1D array of the true labels.
#     groups (np.ndarray): A 1D array indicating the group for each instance.

#     Returns:
#     float: The mean standard deviation of the false positive to false negative ratios across all labels.
#     """
#     unique_labels = np.unique(labels)
#     unique_groups = np.unique(groups)
    
#     ted_stds = []

#     for label in unique_labels:
#         # List to store the ratio of false positives to false negatives for each group
#         ratios = []
        
#         for group in unique_groups:
#             # Define false positives and false negatives for the current group
#             fp = np.sum((predictions == label) & (labels != label) & (groups == group))
#             fn = np.sum((predictions != label) & (labels == label) & (groups == group))
            
#             # Avoid division by zero
#             if fn == 0:
#                 ratio = float('inf')  # Could also choose to skip this group or set a high value
#             else:
#                 ratio = fp / fn
            
#             ratios.append(ratio)
        
#         # Compute the standard deviation of these ratios for the current label
#         std_dev = np.std(ratios) if len(ratios) > 1 else 0
#         ted_stds.append(std_dev)
    
#     # Calculate the mean of the standard deviations across all labels
#     delta_ted_mean = np.mean(ted_stds) if ted_stds else 0
#     return delta_ted_mean


