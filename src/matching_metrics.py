import numpy as np


def match2tpr(matches):

    gt_matches = matches[0]
    gt_unmatched = matches[2]

    n_gt_matched = len(gt_matches)
    n_gt_unmatched = len(gt_unmatched)

    return (n_gt_matched + 1) / (n_gt_matched + n_gt_unmatched + 1)


def match2tp(matches):

    gt_matches = matches[0]
    gt_unmatched = matches[2]

    n_gt_matched = len(gt_matches)
    n_gt_unmatched = len(gt_unmatched)

    return n_gt_matched


def match2fnr(matches):

    pred_matches = matches[1]
    pred_unmatched = matches[3]

    n_pred_matched = len(pred_matches)
    n_pred_unmatched = len(pred_unmatched)

    return n_pred_unmatched / (n_pred_matched + n_pred_unmatched + 1)


def match2gtCoverageRate(matches):

    coverage_binarization_threshold = 0.7

    gt_matches = matches[0]
    gt_unmatched = matches[2]

    binary_coverage_instance_list = []
    for gt_instance_matching in gt_matches:
        instance = gt_instance_matching[0]
        predictions = gt_instance_matching[1]

        instance_coverage = np.sum(instance&np.sum(predictions, axis=0, dtype=bool))/np.sum(instance)
        binary_coverage_instance_list.append(instance_coverage>coverage_binarization_threshold)

    n_gt_matched = len(gt_matches)
    n_gt_unmatched = len(gt_unmatched)


    return np.sum(binary_coverage_instance_list) / (n_gt_matched + n_gt_unmatched + 1)

def match2predCoverageRate(matches):

    coverage_binarization_threshold = 0.3

    pred_matches   = matches[1]
    pred_unmatched = matches[3]

    binary_coverage_instance_list = []
    for pred_instance_matching in pred_matches:
        instance    = pred_instance_matching[0]
        gts = pred_instance_matching[1]

        instance_coverage = np.sum(instance&np.sum(gts, axis=0, dtype=bool))/np.sum(instance)
        binary_coverage_instance_list.append(instance_coverage>coverage_binarization_threshold)

    n_pred_matched = len(pred_matches)
    n_pred_unmatched = len(pred_unmatched)


    return np.sum(binary_coverage_instance_list) / (n_pred_matched + n_pred_unmatched + 1)