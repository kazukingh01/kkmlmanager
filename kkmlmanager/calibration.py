from typing import List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
# local package
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "MockCalibrater",
    "Calibrater",
    "calibration_curve_plot",
]


class MockCalibrater(BaseEstimator):
    def __init__(self, *args, classes: np.ndarray=None, **kwargs):
        assert isinstance(classes, np.ndarray)
        super().__init__(*args, **kwargs)
        self.classes_ = classes
    def __str__(self):
        return __class__.__name__
    def fit(self, *args, **kwargs):
        return self
    def predict(self, *args, X=None, **kwargs):
        return X
    def predict_proba(self, *args, X=None, **kwargs):
        """
        In scikit-learn calibrater input data by "pred_method(X=X)".
        So parameter "X" name is important.
        """
        return X


class Calibrater:
    def __init__(self, model, is_fit_by_class: bool=True):
        logger.info("START")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert isinstance(is_fit_by_class, bool)
        self.model      = model
        self.mock       = None
        self.calibrater = None
        self.is_fit_by_class = is_fit_by_class
        logger.info("END")

    def __str__(self):
        return str(self.calibrater)

    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, **kwargs):
        """
        'input_x' is Probability. is not Features.
        """
        logger.info("START")
        classes = self.model.classes_ if hasattr(self.model, "classes_") else np.sort(np.unique(input_y))
        self.mock       = MockCalibrater(classes=classes)
        self.calibrater = CalibratedClassifierCV(self.mock, cv="prefit", method='isotonic')
        if self.is_fit_by_class:
            self.calibrater.fit(input_x, input_y, *args, **kwargs)
        else:
            input_x = input_x.copy().reshape(-1)
            input_y = np.eye(classes)[input_y.astype(int)].reshape(-1)
            self.calibrater.fit(input_x, input_y, *args, **kwargs)
        logger.info("END")
    
    def predict(self, input_x, *args, **kwargs):
        """
        'input_x' is Features. is not Probability.
        """
        logger.info("START")
        output = self.model.predict(input_x, *args, **kwargs)
        if self.is_fit_by_class:
            output = self.calibrater.predict(output)
        else:
            shape  = output.shape[-1]
            output = self.calibrater.predict(output.reshape(-1))
            output = output.reshape(-1, shape)
        logger.info("END")
        return output

    def predict_proba(self, input_x, *args, **kwargs):
        """
        'input_x' is Features. is not Probability.
        """
        logger.info("START")
        output = self.model.predict_proba(input_x, *args, **kwargs)
        if self.is_fit_by_class:
            output = self.calibrater.predict_proba(output)
        else:
            shape  = output.shape[-1]
            output = self.calibrater.predict_proba(output.reshape(-1))
            output = output.reshape(-1, shape)
        logger.info("END")
        return output


def calibration_curve_plot(prob_pre: np.ndarray, prob_aft: np.ndarray, target: np.ndarray, n_bins: int=10, figsize: tuple=(12, 8)):
    """
    Plot calibration curve.
    Params::
        input_x: output model prediction. before calibration score.
        input_y: label
    """
    logger.info("START")
    assert isinstance(prob_pre, np.ndarray)
    assert isinstance(prob_aft, np.ndarray)
    assert isinstance(target,   np.ndarray)
    assert prob_pre.shape[0] == prob_aft.shape[0] == target.shape[0]
    assert len(prob_pre.shape) == len(prob_aft.shape) == 2
    assert len(target.shape) == 1
    assert target.dtype in [np.int8, np.int16, np.int32, np.int64]
    assert isinstance(n_bins, int) and n_bins >= 1
    classes  = np.arange(prob_pre.shape[1], dtype=int)
    dict_fig = {}
    for i_class in classes:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        fraction_of_positives_pre, mean_predicted_value_pre = calibration_curve((target == i_class), prob_pre[:, i_class], n_bins=n_bins)
        fraction_of_positives_aft, mean_predicted_value_aft = calibration_curve((target == i_class), prob_aft[:, i_class], n_bins=n_bins)
        ax1.plot(mean_predicted_value_pre, fraction_of_positives_pre, "s:", label=f"pre_label_{i_class}")
        ax1.plot(mean_predicted_value_aft, fraction_of_positives_aft, "s-", label=f"aft_label_{i_class}")
        ax2.hist(prob_pre[:, i_class], range=(0, 1), bins=n_bins, label=f"pre_label_{i_class}", histtype="step", lw=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.legend()
        ax2.legend()
        ax1.set_title('Calibration plots (reliability curve)')
        ax2.set_xlabel('Mean predicted value')
        ax1.set_ylabel('Fraction of positives')
        ax2.set_ylabel('Count')
        dict_fig[f"fig_{i_class}"] = fig
    logger.info("END")
    return dict_fig