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
    def __init__(self, model, func_predict: str, is_fit_by_class: bool=True):
        logger.info("START")
        assert isinstance(func_predict, str)
        assert hasattr(model, func_predict)
        assert isinstance(is_fit_by_class, bool)
        logger.info(f"model: {model}, is_fit_by_class: {is_fit_by_class}")
        self.model      = model
        self.mock       = None
        self.calibrater = None
        self.is_fit_by_class = is_fit_by_class
        self.func_predict    = func_predict
        setattr(
            self, func_predict, 
            lambda input_x, *args, is_mock: bool=False, funcname: str="predict_proba", **kwargs: self.predict_common(input_x, *args, is_mock=is_mock, funcname=funcname, **kwargs)
        )
        logger.info("END")

    def __str__(self):
        return str(self.calibrater)

    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, **kwargs):
        """
        'input_x' is Probability. is not Features.
        """
        logger.info("START")
        classes = self.model.classes_ if hasattr(self.model, "classes_") else np.sort(np.unique(input_y))
        assert classes.shape[0] == (classes.max() + 1)
        if self.is_fit_by_class:
            self.mock = MockCalibrater(classes=classes)
        else:
            self.mock = MockCalibrater(classes=np.array([0, 1]))
        self.calibrater = CalibratedClassifierCV(self.mock, cv="prefit", method='isotonic')
        if self.is_fit_by_class:
            self.calibrater.fit(input_x, input_y, *args, **kwargs)
        else:
            input_x = self.to_binary_shape(input_x.copy())
            input_y = np.eye(classes.shape[0])[input_y.astype(int)].reshape(-1)
            self.calibrater.fit(input_x, input_y, *args, **kwargs)
        logger.info("END")
    
    @classmethod
    def to_binary_shape(cls, ndf: np.ndarray):
        ndf = ndf.reshape(-1)
        ndf = np.stack([1 - ndf, ndf]).T
        return ndf
    
    def predict_common(self, input_x, *args, is_mock: bool=False, funcname: str="predict_proba", **kwargs):
        """
        'input_x' is Features. is not Probability ( If is_mock == False ).
        """
        logger.info("START")
        logger.info(f"is_mock: {is_mock}, funcname: {funcname}")
        if is_mock:
            output = input_x
        else:
            output = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
        logger.info(f"predict mode. is_fit_by_class: {self.is_fit_by_class}")
        if self.is_fit_by_class:
            output = getattr(self.calibrater, funcname)(output)
        else:
            shape  = output.shape[-1]
            output = self.to_binary_shape(output)
            output = getattr(self.calibrater, funcname)(output)
            output = output[:, -1].reshape(-1, shape)
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