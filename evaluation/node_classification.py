from sklearn.metrics import accuracy_score
import numpy as np


def delta_std_sp(predictions, labels, groups):
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
    delta_dp = 2 * np.mean(label_group_probs)
    return delta_dp


def delta_max_sp(predictions, labels, groups):
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


def delta_std_eop(predictions, labels, groups):
    """
    Calculate the ΔEO (Delta Equalized Opportunity) metric for multiclass classification. 
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
    mean_eo_std_dev = 2 * np.mean(eo_std_devs)
    return mean_eo_std_dev


def delta_max_eop(predictions, labels, groups):
    """
    Calculate the ΔEO (Delta Equalized Opportunity) metric for multiclass classification. 
    This metric measures the maximum difference in the conditional probabilities 
    P(Ŷ = y | Y = y, S = s) between each pair of sensitive groups for each label y.
    The final metric is the mean of these maximum differences across all labels.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The mean of the maximum differences in conditional probabilities across all pairs of sensitive groups for each label.

    Notes:
    - ΔEO measures fairness in terms of equalized opportunity across sensitive groups.
    - It reflects how consistently the model predicts each class across different groups.
    - Lower values indicate more fairness, suggesting similar true positive rates across groups.
    """
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    max_diffs = []

    # Compute the maximum difference in conditional probabilities P(Ŷ = y | Y = y, S = s) for each label
    for label in unique_labels:
        conditional_probs = []

        # Calculate the conditional probabilities for the current label within each group
        for group in unique_groups:
            true_label_indices = (labels == label) & (groups == group)
            true_label_group_count = np.sum(true_label_indices)
            
            # Avoid division by zero
            if true_label_group_count == 0:
                continue

            # Calculate the conditional probability P(Y_hat=y | Y=y, S=s) for the current group
            correct_pred_for_group = np.sum((predictions[true_label_indices] == label))
            conditional_prob = correct_pred_for_group / true_label_group_count
            conditional_probs.append(conditional_prob)
        
        # Calculate the maximum difference in conditional probabilities for the current label across all group pairs
        max_diff = max([abs(p1 - p2) for p1 in conditional_probs for p2 in conditional_probs])
        max_diffs.append(max_diff)
    
    # Calculate the mean of the maximum differences across all labels
    delta_max_diff = np.mean(max_diffs)
    return delta_max_diff


def delta_std_acc(predictions, labels, groups):
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
    sigma_acc = 2 * np.sqrt(np.mean((accuracies - mean_accuracy) ** 2))
    return sigma_acc


def delta_max_acc(predictions, labels, groups):
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


def delta_std_eod(predictions, labels, groups):
    """
    Calculate the ΔEO (Delta Equalized Odds) metric for multiclass classification. 
    This metric averages the standard deviation of the conditional probabilities 
    P(Ŷ = y | Y = y, S = s) and P(Ŷ = y | Y != y, S = s), respectively, for each label y across all sensitive subgroups s.
    Then average the two standard deviations.

    Parameters:
    predictions (np.ndarray): A 1D numpy array of predicted labels from the model.
    labels (np.ndarray): A 1D numpy array of the ground truth labels.
    groups (np.ndarray): A 1D numpy array indicating the sensitive group for each instance.

    Returns:
    float: The ΔEO metric, representing the mean of the standard deviations of the 
           conditional probabilities across sensitive groups for each label.

    Notes:
    - ΔEO measures fairness in terms of equalized odds across sensitive groups.
    - It reflects how consistently the model predicts each class across different groups.
    - Lower values indicate more fairness, suggesting similar true positive rates across groups.
    """
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    eod_metrics = []

    for label in unique_labels:
        conditional_probs_correct = []  # For P(Ŷ = y | Y = y, S = s)
        conditional_probs_incorrect = []  # For P(Ŷ = y | Y ≠ y, S = s)

        for group in unique_groups:
            # Indices for the current group
            group_indices = (groups == group)
            
            # Correct scenario: Y = y
            correct_indices = (labels == label) & group_indices
            correct_count = np.sum(correct_indices)
            if correct_count > 0:
                correct_predictions = np.sum((predictions[correct_indices] == label))
                conditional_prob_correct = correct_predictions / correct_count
                conditional_probs_correct.append(conditional_prob_correct)
            
            # Incorrect scenario: Y ≠ y
            incorrect_indices = (labels != label) & group_indices
            incorrect_count = np.sum(incorrect_indices)
            if incorrect_count > 0:
                incorrect_predictions = np.sum((predictions[incorrect_indices] == label))
                conditional_prob_incorrect = incorrect_predictions / incorrect_count
                conditional_probs_incorrect.append(conditional_prob_incorrect)
        # Calculate the standard deviations for correct and incorrect predictions
        std_dev_correct = np.std(conditional_probs_correct) if len(conditional_probs_correct) > 1 else 0
        std_dev_incorrect = np.std(conditional_probs_incorrect) if len(conditional_probs_incorrect) > 1 else 0

        # Average the two standard deviations for the current label
        mean_std_dev = np.mean([std_dev_correct, std_dev_incorrect])
        eod_metrics.append(mean_std_dev)
    # Calculate the mean of the ΔEOD metrics across all labels
    mean_eod = np.mean(eod_metrics)
    return mean_eod


def delta_max_eod(predictions, labels, groups):
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    max_diffs_avg = []

    for label in unique_labels:
        conditional_probs_correct = []  # For P(Ŷ = y | Y = y, S = s)
        conditional_probs_incorrect = []  # For P(Ŷ = y | Y ≠ y, S = s)

        for group in unique_groups:
            # Indices for the current group
            group_indices = (groups == group)
            
            # Correct scenario: Y = y
            correct_indices = (labels == label) & group_indices
            correct_count = np.sum(correct_indices)
            if correct_count > 0:
                correct_predictions = np.sum((predictions[correct_indices] == label))
                conditional_prob_correct = correct_predictions / correct_count
                conditional_probs_correct.append(conditional_prob_correct)
            
            # Incorrect scenario: Y ≠ y
            incorrect_indices = (labels != label) & group_indices
            incorrect_count = np.sum(incorrect_indices)
            if incorrect_count > 0:
                incorrect_predictions = np.sum((predictions[incorrect_indices] == label))
                conditional_prob_incorrect = incorrect_predictions / incorrect_count
                conditional_probs_incorrect.append(conditional_prob_incorrect)

        # Calculate the maximum differences for correct and incorrect scenarios
        max_diff_correct = max([abs(p1 - p2) for p1 in conditional_probs_correct for p2 in conditional_probs_correct]) if len(conditional_probs_correct) > 1 else 0
        max_diff_incorrect = max([abs(p1 - p2) for p1 in conditional_probs_incorrect for p2 in conditional_probs_incorrect]) if len(conditional_probs_incorrect) > 1 else 0

        # Average the two maximum differences for the current label
        mean_max_diff = np.mean([max_diff_correct, max_diff_incorrect])
        max_diffs_avg.append(mean_max_diff)

    # Calculate the mean of the averaged maximum differences across all labels
    delta_max_eod_metric = np.mean(max_diffs_avg)
    return delta_max_eod_metric
