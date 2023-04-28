from typing import List
import numpy as np
from functools import partial
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
        In scikit-learn, calibrater has api which is like "pred_method(X=X)".
        So "X" is important api name.
        """
        return X


class Calibrater:
    def __init__(self, model, func_predict: str, is_normalize: bool=False, is_reg: bool=False):
        logger.info("START")
        assert isinstance(func_predict, str)
        assert hasattr(model, func_predict)
        assert isinstance(is_normalize, bool)
        assert isinstance(is_reg, bool)
        logger.info(f"model: {model}, func_predict: {func_predict}, is_normalize: {is_normalize}, is_reg: {is_reg}")
        self.model        = model
        self.mock         = None
        self.calibrater   = None
        self.func_predict = func_predict
        self.is_normalize = is_normalize
        self.is_reg       = is_reg
        setattr(
            self, func_predict, partial(self.predict_common, is_mock=False)
        )
        logger.info("END")

    def __str__(self):
        return str(self.calibrater)

    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, n_bins: int=10, **kwargs):
        """
        'input_x' must be probabilities, not Features.
        """
        logger.info("START")
        assert isinstance(input_x, np.ndarray) and len(input_x.shape) == 2 and input_x.shape[-1] >= 2
        assert isinstance(input_y, np.ndarray) and len(input_y.shape) == 1
        assert isinstance(n_bins, int) and n_bins >= 5
        if self.is_reg:
            self.classes_ = np.arange(1, dtype=int) if len(input_y.shape) == 1 else np.arange(input_y.shape[-1], dtype=int)
        else:
            self.classes_ = np.sort(np.unique(input_y))
        self.mock       = MockCalibrater(classes=self.classes_)
        self.calibrater = CalibratedClassifierCV(self.mock, cv="prefit", method='isotonic')
        self.calibrater.fit(input_x, input_y, *args, **kwargs)
        output          = self.predict_common(input_x, is_mock=True)
        self.calib_fig  = calibration_curve_plot(input_x, output, input_y, n_bins=n_bins)
        logger.info("END")
        
    def predict_common(self, input_x, *args, is_mock: bool=False, **kwargs):
        """
        Note::
            If is_mock == False, "input_x" must be features.
            If is_mock == True,  "input_x" must be probabilities.
        """
        logger.info(f"START {self.__class__}")
        logger.info(f"is_mock: {is_mock}, model predict: {self.func_predict}, is_normalize: {self.is_normalize}")
        if is_mock:
            output = input_x
        else:
            output = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
            if len(output.shape) == 1:
                # You must be careful when model is trained by binary logloss
                # Calibration require the shape has more than 2 even if output's shape has only 1 because of being trained by binary.
                output = np.stack([1 - output, output]).T
        funcname = "predict" if self.is_reg else "predict_proba"
        output   = getattr(self.calibrater, funcname)(output)
        if self.is_normalize:
            output = (output / output.sum(axis=-1).reshape(-1, 1))
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