import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
# import pylab as plt

from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE

sys.path.append(os.path.abspath('../src/'))

from dpipe_metrics import get_matching
from dpipe_metrics import hausdorff_distance, surface_distances, dice_score, assd, iou

metrics_dict = {
    "dice_coefficient": {"2d": dice_score, "3d": dice_score},
#     "mae": {"2d": lambda x, y: np.abs(x - y).mean(), "3d": lambda x, y: np.abs(x - y).mean()},
#     "mse": {"2d": lambda x, y: ((x - y) ** 2).mean(), "3d": lambda x, y: ((x - y) ** 2).mean()},
    "hausdorff_distance": {"2d": hausdorff_distance, "3d": hausdorff_distance},
    "surface_distances": {"2d": surface_distances, "3d": surface_distances},
    "assd": {"2d": assd, "3d": assd}
}

unary_metrics_dict = {
    "area": {"2d": lambda x: (x > 0).sum(), "3d": lambda x: (x > 0).sum()}
}


class BaseQualityEstimator(BaseEstimator, ClassifierMixin):
    """Base Estimator for segmentation quality assessment"""

    def __init__(self, metrics=["dice_coefficient"], unary_metrics=["area"], meta_clf=LGBMClassifier()):
        """
        Args:
            metrics: list of strings: metrics to be computed on pairs of preds and gt
            unary_metrics: list of string: metrics to be computed on preds directly
        
        TODO: params??
        """
        self.meta_clf = meta_clf
        self.metrics = list(filter(lambda _: _ in metrics_dict, metrics))
        self.unary_metrics = list(filter(lambda _: _ in unary_metrics_dict, unary_metrics))
        
        self.data_type = "3d"
        self.X_metrics = None
    
    
    def fit(self, X, Xy=None, y=None):
        """
        
        """
        assert len(X) == len(Xy) == len(y)
        # get the dimensionality of the data
#         self.data_type = self._check_data_type(X)
        # compute all the metrics on the pairs from X (predictions) and Xy (gt)
        self.X_metrics = self._compute_metrics(X, Xy)
        # fit meta-classifier to metrics and human-made labels
        self.meta_clf.fit(self.X_metrics, y)

        return self
    
    def predict(self, X, Xy):
        
        X_metrics = self._compute_metrics(X, Xy)
        
        y_pred = self.meta_clf.predict(X_metrics)
        
        return y_pred
    
    def predict_proba(self, X, Xy):
        
        X_metrics = self._compute_metrics(X, Xy)
        
        y_pred = self.meta_clf.predict_proba(X_metrics)
        
        return y_pred
    
    def _compute_metrics(self, X, Xy):
        
        def _metrics(x, xy):
            metrics_computed = dict()
            for metric_ in self.metrics:
                metrics_computed[metric_] = metrics_dict[metric_][self.data_type](x, xy)
            return metrics_computed
        
        def _unary_metrics(x):
            unary_metrics_computed = dict()
            for metric_ in self.unary_metrics:
                unary_metrics_computed[metric_] = unary_metrics_dict[metric_][self.data_type](x)
            
            return unary_metrics_computed
        
        def _matching_metrics(x_decomp, xy_decomp):
            matching = get_matching(x_decomp, xy_decomp, metric=iou, metric_ths=[0.1])

            def _match2tpr(matches):

                gt_matches = matches[0]
                gt_unmatched = matches[2]

                n_gt_matched = len(gt_matches)
                n_gt_unmatched = len(gt_unmatched)

                return (n_gt_matched + 1) / (n_gt_matched + n_gt_unmatched + 1)
            
            
            def _match2tp(matches):

                gt_matches = matches[0]
                gt_unmatched = matches[2]

                n_gt_matched = len(gt_matches)
                n_gt_unmatched = len(gt_unmatched)

                return n_gt_matched
        

            def _match2fnr(matches):

                pred_matches = matches[1]
                pred_unmatched = matches[3]

                n_pred_matched = len(pred_matches)
                n_pred_unmatched = len(pred_unmatched)

                return n_pred_unmatched / (n_pred_matched + n_pred_unmatched + 1)
            
            
            def _match2gtCoverageRate(matches):
                
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
            
            def _match2predCoverageRate(matches):
                
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
            
            tpr = _match2tpr(matching[0])
            fnr = _match2fnr(matching[0])
            tp = _match2tp(matching[0])
            gtCovThrRate   = _match2gtCoverageRate(matching[0])
            predCovThrRate = _match2predCoverageRate(matching[0])
            return {#"tpr": tpr, 
                    #"fnr": fnr, 
                    "tp": tp,
                    "gtCovThrRate": gtCovThrRate, 
                    "predCovThrRate": predCovThrRate}
        
        metrics_computed = []
        
        for x_, xy_ in zip(X, Xy):
            metrics_temp_ = _metrics(x_[0], xy_[0])
            metrics_temp_.update(_unary_metrics(x_[0]))
            matching_metrics_temp_ = _matching_metrics(x_[1:], xy_[1:])
            metrics_temp_.update(matching_metrics_temp_)
            metrics_computed.append(metrics_temp_)
            
        df_metrics_computed = pd.DataFrame(metrics_computed)
        
        return df_metrics_computed
        
    def _check_data_type(self, X):
        """
        TODO:
        """
        # заглушка:
        if len(X.shape) == 2:
            return "2d"
        elif X.shape[2] == 1:
            return "2d"
        else:
            return "3d"

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X))) 