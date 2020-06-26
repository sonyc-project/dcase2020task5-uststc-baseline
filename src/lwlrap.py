"""
This label-weighted label-ranking average precision (lwlrap) comes from
Dan Ellis's implementation for DCASE 2019 Task 2:

https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8
"""
import numpy as np
import oyaml as yaml
from annotations import parse_ground_truth, parse_fine_prediction, \
    parse_coarse_prediction


def _one_sample_positive_class_precisions(truth, scores):
    """
    Calculate precisions for each true class for a single sample.

    Parameters
    ----------
        truth: array of bool, shape = [num_classes,]
            array of bools indicating which classes are true.
        scores: array of float, shape = [num_classes,]
            array of individual classifier scores.

    Returns
    -------
        pos_class_indices: array of int, shape = [num_true_classes,]
            array of indices of the true classes for this sample.
        pos_class_precisions: array of int, shape = [num_classes,]
            precisions corresponding to each of those classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
         return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
         retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
         (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculate label-weighted label-ranking average precision per class.

    Parameters
    ----------
        truth: array of bool, shape = [num_samples, num_classes]
            array of boolean ground-truth of presence of each class
            for each sample.
        scores: array of float, shape = [num_samples, num_classes]
            array of the classifier-under-test's real-valued score
            for each class for each sample.

    Returns
    -------
        per_class_lwlrap: array of float, shape = [num_classes,]
            array of lwlrap for each class.
        weight_per_class: array of float, shape = [num_classes,]
            giving the prior of each class within the truth labels.
            Then the overall unbalanced lwlrap is simply
            np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(truth[sample_num, :],
                                                  scores[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))

    return per_class_lwlrap, weight_per_class


def calculate_lwlrap_metrics(prediction_path, annotation_path, yaml_path, mode):
    """Compute lwlrap metrics from predictions and ground truth annotations."""
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.Loader)

    # Parse ground truth.
    gt_df = parse_ground_truth(annotation_path, yaml_path)

    # Parse predictions, tag ids, and tag names.
    if mode == "fine":
        pred_df = parse_fine_prediction(prediction_path, yaml_path)
        tag_id_list = []
        tag_name_list = []
        for coarse_id, fine_dict in yaml_dict["fine"].items():
            for fine_id, fine_name in fine_dict.keys():
                coarse_fine_id = "{}-{}".format(coarse_id, fine_id)
                tag_id_list.append(coarse_fine_id)
                tag_name_list.append(fine_name)
    elif mode == "coarse":
        pred_df = parse_coarse_prediction(prediction_path, yaml_path)
        tag_id_list, tag_name_list = zip(*yaml_dict["coarse"].items())
        tag_id_list, tag_name_list = list(tag_id_list), list(tag_name_list)

    # Convert dataframes into numpy arrays
    truth = gt_df[tag_id_list].to_numpy()
    scores = pred_df[tag_id_list].to_numpy()

    # Compute lwlrap
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    overall_lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    # Store per-class information in dictionaries
    class_lwlrap_dict = dict(zip(tag_name_list, per_class_lwlrap))
    class_weight_dict = dict(zip(tag_name_list, weight_per_class))

    return overall_lwlrap, class_lwlrap_dict, class_weight_dict

