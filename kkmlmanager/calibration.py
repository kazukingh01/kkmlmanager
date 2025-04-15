import copy, tqdm
import numpy as np
from functools import partial
from sklearn.calibration import calibration_curve
from sklearn.calibration import label_binarize
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from kklogger import set_logger

# local package
from .util.numpy import NdarrayWithErr, nperr_stack
from .util.com import encode_object, decode_object
LOGGER = set_logger(__name__)


__all__ = [
    "IsotonicRegressionWithError",
    "calibration_curve_plot",
]


class IsotonicRegressionWithError:
    def __init__(self, y_min: float | None=None, y_max: float | None=None, increasing: bool=True, set_first_score: bool=False):
        assert isinstance(y_min, (float, type(None)))
        assert isinstance(y_max, (float, type(None)))
        assert isinstance(increasing, bool)
        assert isinstance(set_first_score, bool)
        if set_first_score and increasing:
            assert y_min is None
        if set_first_score and not increasing:
            assert y_max is None
        if y_min is None: y_min = 0.0
        if y_max is None: y_max = 1.0
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=increasing, out_of_bounds="clip")
        self.set_first_score = set_first_score
        self.map_x     = np.array([], dtype=float)
        self.map_n_bin = np.array([], dtype=float)
        self.map_err   = np.array([], dtype=float)
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model}, map_x={self.map_x[:3]} ..., map_n_bin={self.map_n_bin[:3]} ..., map_err={self.map_err[:3]} ..., set_first_score={self.set_first_score})"
    def __repr__(self):
        return self.__str__()
    def fit(self, X: np.ndarray, Y: np.ndarray, n_bootstrap: int=100):
        LOGGER.info("START")
        assert isinstance(X, np.ndarray) and len(X.shape) == 1
        assert isinstance(Y, np.ndarray) and len(Y.shape) == 1
        assert X.shape == Y.shape
        assert Y.dtype in [int, np.int8, np.int16, np.int32, np.int64]
        assert Y.max() == 1 and Y.min() == 0
        if   self.set_first_score and self.model.increasing:
            self.model.y_min = X.min()
        elif self.set_first_score and not self.model.increasing:
            self.model.y_max = X.max()
        model_base = copy.deepcopy(self.model)
        self.model.fit(X, Y)
        self.map_x     = np.array([self.model.X_thresholds_[0] / 2.0] + ((self.model.X_thresholds_[:-1] + self.model.X_thresholds_[1:]) / 2.0).tolist() + [self.model.X_thresholds_[-1] * 2.0])
        self.map_n_bin = np.histogram2d(X, Y, bins=[self.map_x, [-0.5,0.5,1.5]])[0]
        # bootstrap sampling
        ndf_idx = np.random.randint(0, X.shape[0], (n_bootstrap, X.shape[0]))
        ndf_val = []
        for ndfwk in tqdm.tqdm(ndf_idx):
            model = copy.deepcopy(model_base).fit(X[ndfwk], Y[ndfwk])
            ndf_val.append(model.predict(self.map_x))
        self.map_err = np.array(ndf_val).std(axis=0)
        LOGGER.info("END")
        return self
    def predict(self, X: np.ndarray):
        LOGGER.info("START")
        assert isinstance(X, (list, np.ndarray))
        if isinstance(X, np.ndarray):
            assert len(X.shape) == 1
        else:
            assert isinstance(X, list)
        val = self.model.predict(X)
        err = self.map_err[np.digitize(X, self.model.X_thresholds_)]
        LOGGER.info("END")
        return NdarrayWithErr(val, err)


class MultiLabelRegressionWithError:
    def __init__(self, increasing: bool=True, set_first_score: bool=True):
        LOGGER.info("START")
        assert isinstance(increasing, bool)
        assert isinstance(set_first_score, bool)
        self.basemodel = partial(IsotonicRegressionWithError, increasing=increasing, set_first_score=set_first_score)
        self.list_models: list[IsotonicRegressionWithError] = []
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(basemodel={self.basemodel}, list_models={len(self.list_models)})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self) -> dict:
        return {
            "increasing": self.basemodel.keywords["increasing"],
            "set_first_score": self.basemodel.keywords["set_first_score"],
            "list_models": [encode_object(x) for x in self.list_models]
        }
    @classmethod
    def from_dict(cls, dict_model: dict):
        assert isinstance(dict_model, dict)
        ins = cls(increasing=dict_model["increasing"], set_first_score=dict_model["set_first_score"])
        ins.list_models = [decode_object(x) for x in dict_model["list_models"]]
        return ins
    def fit(self, input_x: np.ndarray, input_y: np.ndarray, is_reg: bool=False, n_bootstrap: int=100):
        LOGGER.info("START")
        assert isinstance(input_x, np.ndarray) and len(input_x.shape) in [1,2]
        assert isinstance(input_y, np.ndarray) and len(input_y.shape) in [1,2]
        assert input_x.shape[0] == input_y.shape[0]
        assert isinstance(is_reg, bool)
        if is_reg:
            assert input_y.dtype in [np.float16, np.float32, np.float64, np.float128, float]
        else:
            assert input_y.dtype in [np.int8, np.int16, np.int32, np.int64, int, str, np.dtypes.StrDType]
            """
            >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
            array( [[1],
                    [0],
                    [0],
                    [1]])
            >>> input_y = np.array([0,1,1])
            >>> label_binarize(input_y, classes=np.sort(np.unique(input_y)))
            array( [[0],
                    [1],
                    [1]])
            >>> input_y = np.array([0,1,1,2])
            >>> label_binarize(input_y, classes=np.sort(np.unique(input_y)))
            array( [[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
            """
            if input_x.shape != input_y.shape:
                input_y = label_binarize(input_y, classes=np.sort(np.unique(input_y)))
        assert input_x.shape == input_y.shape
        if len(input_x.shape) == 1: input_x = input_x.reshape(-1, 1)
        if len(input_y.shape) == 1: input_y = input_y.reshape(-1, 1)
        for i, (_input_x, _input_y) in enumerate(zip(input_x.T, input_y.T)):
            LOGGER.info(f"fitting ... {i + 1} / {input_y.shape[-1]}")
            self.list_models.append(self.basemodel())
            self.list_models[-1].fit(_input_x, _input_y, n_bootstrap=n_bootstrap)
        LOGGER.info("END")
        return self
    def predict(self, input_x: np.ndarray) -> NdarrayWithErr:
        LOGGER.info("START")
        assert isinstance(input_x, np.ndarray) and len(input_x.shape) in [1,2]
        if len(input_x.shape) == 1: input_x = input_x.reshape(-1, 1)
        list_pred = []
        for _input_x, model in zip(input_x.T, self.list_models):
            list_pred.append(model.predict(_input_x))
        LOGGER.info("END")
        return nperr_stack(list_pred, dtype=np.float64).T


def calib_with_error(prob: np.ndarray, target: np.ndarray, n_bins: int=10):
    assert isinstance(prob, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert prob.shape == target.shape
    assert len(prob.shape) == len(target.shape) == 1
    assert target.dtype in [np.int8, np.int16, np.int32, np.int64, int, bool, np.bool_]
    target = target.astype(int)
    assert np.sort(np.unique(target)).tolist() == [0, 1]
    assert isinstance(n_bins, int) and n_bins >= 2
    ndf_bin = np.array([-1] + np.linspace(0, 1, n_bins).tolist()[1:-1] + [2], dtype=float)
    ndf_n   = np.histogram2d(prob, target, bins=[ndf_bin, [-0.5,0.5,1.5]])[0]
    ndf_idx = np.digitize(prob, ndf_bin)
    ndf_x   = np.array([prob[ndf_idx == i].mean() for i in np.arange(ndf_bin.shape[0] + 1)[1:-1]])
    ndf_x_e = np.array([prob[ndf_idx == i].std()  for i in np.arange(ndf_bin.shape[0] + 1)[1:-1]])
    ndf_y   = ndf_n[:, -1] / ndf_n.sum(axis=-1)
    ndf_y_e = (ndf_n[:, 0] * ndf_n[:, 1]) / (ndf_n.sum(axis=-1) ** 2) * np.sqrt((1.0 / ndf_n[:, 0]) + (1.0 / ndf_n[:, 1]))
    return ndf_x, ndf_x_e, ndf_y, ndf_y_e


def calibration_curve_plot(prob_pre: np.ndarray, prob_aft: np.ndarray, target: np.ndarray, n_bins: int=10, figsize: tuple=(12, 8)):
    """
    Plot calibration curve.
    Params::
        input_x: output model prediction. before calibration score.
        input_y: label
    """
    LOGGER.info("START")
    assert isinstance(prob_pre, np.ndarray)
    assert isinstance(prob_aft, (np.ndarray, NdarrayWithErr))
    assert isinstance(target,   np.ndarray)
    assert prob_pre.shape[0] == prob_aft.shape[0] == target.shape[0]
    assert len(prob_pre.shape) == len(prob_aft.shape) == 2
    assert len(target.shape) == 1
    assert target.dtype in [np.int8, np.int16, np.int32, np.int64] # For classification
    assert isinstance(n_bins, int) and n_bins >= 1
    classes  = np.arange(prob_pre.shape[1], dtype=int)
    assert np.allclose(np.sort(np.unique(target)).astype(int), classes)
    dict_fig = {}
    for i_class in classes:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ndf_x, ndf_x_e, ndf_y, ndf_y_e = calib_with_error(prob_pre[:, i_class], (target == i_class), n_bins=n_bins)
        ax1.errorbar(ndf_x, ndf_y, xerr=ndf_x_e, yerr=ndf_y_e, fmt='s:', capsize=2, label=f"pre_label_{i_class}")
        ndf_x, ndf_x_e, ndf_y, ndf_y_e = calib_with_error(prob_aft[:, i_class], (target == i_class), n_bins=n_bins)
        ax1.errorbar(ndf_x, ndf_y, xerr=ndf_x_e, yerr=ndf_y_e, fmt='s-', capsize=2, label=f"aft_label_{i_class}")
        ax2.hist(prob_pre[:, i_class], range=(0, 1), bins=n_bins, label=f"pre_label_{i_class}", histtype="step", lw=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.legend()
        ax2.legend()
        ax1.set_title('Calibration plots (reliability curve)')
        ax2.set_xlabel('Mean predicted value')
        ax1.set_ylabel('Fraction of positives')
        ax2.set_ylabel('Count')
        dict_fig[f"fig_{i_class}"] = fig
    LOGGER.info("END")
    return dict_fig