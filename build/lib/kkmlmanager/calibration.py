import json
import numpy as np
from functools import partial
from sklearn.calibration import calibration_curve, label_binarize
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# local package
from kkmlmanager.models import BaseModel
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Calibrater",
    "calibration_curve_plot",
]


class MultiLabelIsotonicRegression:
    def __init__(self):
        logger.info("START")
        self.basemodel = partial(IsotonicRegression, out_of_bounds="clip")
        self.list_models: list[IsotonicRegression] = []
        logger.info("END")
    
    def __str__(self):
        return f"{self.__class__.__name__}(basemodel={self.basemodel}, list_models={len(self.list_models)})"
    
    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, **kwargs):
        logger.info("START")
        assert isinstance(input_x, np.ndarray) and len(input_x.shape) in [1,2]
        assert isinstance(input_y, np.ndarray) and len(input_y.shape) in [1,2]
        is_reg = True if input_y.dtype in [np.float16, np.float32, np.float64, np.float128, float] else False
        if is_reg == False:
            """
            >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
            array( [[1],
                    [0],
                    [0],
                    [1]])
            """
            input_y = label_binarize(input_y, classes=np.sort(np.unique(input_y)))
        if len(input_x.shape) == 1: input_x = input_x.reshape(-1, 1)
        if len(input_y.shape) == 1: input_y = input_y.reshape(-1, 1)
        assert input_x.shape == input_y.shape
        for _input_x, _input_y in zip(input_x.T, input_y.T):
            self.list_models.append(self.basemodel())
            self.list_models[-1].fit(_input_x, _input_y, *args, **kwargs)
        logger.info("END")
    
    def predict(self, input_x: np.ndarray, *args, **kwargs):
        logger.info("START")
        assert isinstance(input_x, np.ndarray) and len(input_x.shape) in [1,2]
        if len(input_x.shape) == 1: input_x = input_x.reshape(-1, 1)
        list_pred = []
        for _input_x, model in zip(input_x.T, self.list_models):
            list_pred.append(model.predict(_input_x, *args, **kwargs))
        logger.info("END")
        return np.stack(list_pred).T


class Calibrater(BaseModel):
    def __init__(self, model, func_predict: str, is_normalize: bool=False, is_reg: bool=False, is_binary_fit: bool=False):
        logger.info("START")
        assert isinstance(func_predict, str)
        assert hasattr(model, func_predict)
        assert isinstance(is_normalize, bool)
        assert isinstance(is_reg, bool)
        assert isinstance(is_binary_fit, bool)
        if is_reg:
            assert is_binary_fit == False
            assert is_normalize  == False
        logger.info(f"model: {model}, func_predict: {func_predict}, is_normalize: {is_normalize}, is_reg: {is_reg}, is_binary_fit: {is_binary_fit}")
        self.model         = model
        self.calibrater    = MultiLabelIsotonicRegression()
        self.func_predict  = func_predict
        self.is_normalize  = is_normalize
        self.is_reg        = is_reg
        self.is_binary_fit = is_binary_fit
        super().__init__(func_predict, is_mock=False)
        logger.info("END")
    
    def to_json(self) -> dict:
        return {
            "func_predict": self.func_predict,
            "is_normalize": self.is_normalize,
            "is_reg": self.is_reg,
            "is_binary_fit": self.is_binary_fit,
            "model": self.model.__class__.__name__,
            f"{self.func_predict}": str(getattr(self, self.func_predict)),
            "calibrater": str(self.calibrater)
        }

    def __str__(self):
        return self.__class__.__name__ + " " + json.dumps(self.to_json(), indent=4)

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
        if self.is_binary_fit:
            logger.info("set for binary fitting.")
            assert input_x.shape[-1] >= 3
            assert self.classes_ == np.arange(input_x.shape[-1], dtype=self.classes_.dtype)
            assert self.is_reg == False
            input_y = np.eye(input_x.shape[-1])[input_y].reshape(-1).astype(int)
            input_x = input_x.reshape(-1, 1)
            self.classes_ = np.array([0, 1])
        self.calibrater.fit(input_x, input_y, *args, **kwargs)
        output = self.predict_common(input_x, is_mock=True)
        if len(output.shape) == 1:
            output     = np.stack([1 - output, output]).T
        self.calib_fig = calibration_curve_plot(input_x, output, input_y, n_bins=n_bins)
        logger.info("END")
        
    def predict_common(self, input_x, *args, is_mock: bool=False, **kwargs):
        """
        Note::
            If is_mock == False, "input_x" must be features.
            If is_mock == True,  "input_x" must be probabilities.
        """
        logger.info(f"START {self.__class__}")
        logger.info(f"is_mock: {is_mock}, model predict function: '{self.func_predict}', is_normalize: {self.is_normalize}, is_binary_fit: {self.is_binary_fit}")
        is_binary = False
        if is_mock:
            output = input_x
        else:
            output = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
            if len(output.shape) == 1:
                output    = output.reshape(-1, 1)
                is_binary = True
        classes = output.shape[-1]
        if self.is_binary_fit:
            logger.info("binary fitting mode.")
            output = output.reshape(-1, 1)
        logger.info("calibrate output ...")
        output = self.calibrater.predict(output)
        if self.is_binary_fit:
            output = output[:, -1]
            output = output.reshape(-1, classes)
        if self.is_normalize:
            logger.info("normalize output ...")
            output = (output / output.sum(axis=-1).reshape(-1, 1))
        if is_binary:
            # return to same shape as model's output
            output = output[:, -1]
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