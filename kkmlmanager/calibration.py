import copy, tqdm
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from functools import partial
from sklearn.calibration import label_binarize
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from kklogger import set_logger

# local package
from .util.numpy import NdarrayWithErr
from .util import numpy as npe
from .util.com import encode_object, decode_object
LOGGER = set_logger(__name__)


__all__ = [
    "IsotonicRegressionWithError",
    "MultiLabelRegressionWithError",
    "TemperatureScaling",
    "calibration_curve_plot",
    "expected_calibration_error",
]


class BaseCalibrator:
    def __init__(self):
        pass
    def __str__(self):
        raise NotImplementedError()
    def to_dict(self) -> dict:
        raise NotImplementedError()
    @classmethod
    def from_dict(cls):
        raise NotImplementedError()
    def fit(self):
        raise NotImplementedError()
    def predict(self) -> np.ndarray | NdarrayWithErr:
        raise NotImplementedError()


class IsotonicRegressionWithError:
    def __init__(self, y_min: float | None=None, y_max: float | None=None, increasing: bool=True, set_first_score: bool=False):
        LOGGER.info("START")
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
        super().__init__()
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model}, map_x={self.map_x[:3]} ..., map_n_bin={self.map_n_bin[:3]} ..., map_err={self.map_err[:3]} ..., set_first_score={self.set_first_score})"
    def __repr__(self):
        return self.__str__()
    def fit(self, probs: np.ndarray, labels: np.ndarray, n_bootstrap: int=100):
        LOGGER.info("START")
        assert isinstance(probs,  np.ndarray) and len(probs. shape) == 1
        assert isinstance(labels, np.ndarray) and len(labels.shape) == 1
        assert probs.shape == labels.shape
        assert labels.dtype in [int, np.int8, np.int16, np.int32, np.int64]
        assert labels.max() == 1 and labels.min() == 0
        if   self.set_first_score and self.model.increasing:
            self.model.y_min = probs.min()
        elif self.set_first_score and not self.model.increasing:
            self.model.y_max = probs.max()
        model_base = copy.deepcopy(self.model)
        self.model.fit(probs, labels)
        self.map_x     = np.array([self.model.X_thresholds_[0] / 2.0] + ((self.model.X_thresholds_[:-1] + self.model.X_thresholds_[1:]) / 2.0).tolist() + [self.model.X_thresholds_[-1] * 2.0])
        self.map_n_bin = np.histogram2d(probs, labels, bins=[self.map_x, [-0.5,0.5,1.5]])[0]
        # bootstrap sampling
        ndf_idx = np.random.randint(0, probs.shape[0], (n_bootstrap, probs.shape[0]))
        ndf_val = []
        for ndfwk in tqdm.tqdm(ndf_idx):
            model = copy.deepcopy(model_base).fit(probs[ndfwk], labels[ndfwk])
            ndf_val.append(model.predict(self.map_x))
        self.map_err = np.array(ndf_val).std(axis=0)
        LOGGER.info("END")
        return self
    def predict(self, probs: list[int | float] | np.ndarray) -> NdarrayWithErr:
        LOGGER.info("START")
        assert isinstance(probs, (list, np.ndarray))
        if isinstance(probs, np.ndarray):
            assert len(probs.shape) == 1
        else:
            assert isinstance(probs, list)
        val = self.model.predict(probs)
        err = self.map_err[np.digitize(probs, self.model.X_thresholds_)]
        LOGGER.info("END")
        return NdarrayWithErr(val, err)


class MultiLabelRegressionWithError(BaseCalibrator):
    def __init__(self, increasing: bool=True, set_first_score: bool=True, is_reg: bool=False):
        LOGGER.info("START")
        assert isinstance(increasing, bool)
        assert isinstance(set_first_score, bool)
        assert isinstance(is_reg, bool)
        self.basemodel = partial(IsotonicRegressionWithError, increasing=increasing, set_first_score=set_first_score)
        self.list_models: list[IsotonicRegressionWithError] = []
        self.is_reg = is_reg
        super().__init__()
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(basemodel={self.basemodel}, list_models={len(self.list_models)}, is_reg={self.is_reg})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self) -> dict:
        return {
            "__BaseModel__": "kkmlmanager.calibration.MultiLabelRegressionWithError",
            "increasing": self.basemodel.keywords["increasing"],
            "set_first_score": self.basemodel.keywords["set_first_score"],
            "is_reg": self.is_reg,
            "list_models": [encode_object(x) for x in self.list_models]
        }
    @classmethod
    def from_dict(cls, dict_model: dict):
        assert isinstance(dict_model, dict)
        ins = cls(increasing=dict_model["increasing"], set_first_score=dict_model["set_first_score"], is_reg=dict_model["is_reg"])
        ins.list_models = [decode_object(x) for x in dict_model["list_models"]]
        return ins
    def fit(self, probs: np.ndarray, labels: np.ndarray, n_bootstrap: int=100):
        LOGGER.info("START")
        assert isinstance(probs,  np.ndarray) and len(probs. shape) in [1,2]
        assert isinstance(labels, np.ndarray) and len(labels.shape) in [1,2]
        assert probs.shape[0] == labels.shape[0]
        if self.is_reg:
            assert labels.dtype in [np.float16, np.float32, np.float64, np.float128, float]
        else:
            assert labels.dtype in [np.int8, np.int16, np.int32, np.int64, int, str, np.dtypes.StrDType]
            """
            >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
            array( [[1],
                    [0],
                    [0],
                    [1]])
            >>> labels = np.array([0,1,1])
            >>> label_binarize(labels, classes=np.sort(np.unique(labels)))
            array( [[0],
                    [1],
                    [1]])
            >>> labels = np.array([0,1,1,2])
            >>> label_binarize(labels, classes=np.sort(np.unique(labels)))
            array( [[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
            """
            classes = np.sort(np.unique(labels))
            if probs.shape != labels.shape:
                labels = label_binarize(labels, classes=classes)
            if classes.shape[0] == 2 and labels.ndim == 2 and labels.shape[1] == 1:
                assert np.allclose(classes, np.array([0, 1]))
        if len(probs. shape) == 1: probs  = probs. reshape(-1, 1)
        if len(labels.shape) == 1: labels = labels.reshape(-1, 1)
        assert probs.shape == labels.shape
        for i, (_probs, _labels) in enumerate(zip(probs.T, labels.T)):
            LOGGER.info(f"fitting ... {i + 1} / {probs.shape[-1]}")
            self.list_models.append(self.basemodel())
            self.list_models[-1].fit(_probs, _labels, n_bootstrap=n_bootstrap)
        LOGGER.info("END")
        return self
    def predict(self, probs: np.ndarray) -> NdarrayWithErr:
        LOGGER.info("START")
        assert isinstance(probs, np.ndarray) and len(probs.shape) in [1,2]
        is_binary = (probs.ndim == 1)
        if len(probs.shape) == 1: probs = probs.reshape(-1, 1)
        assert probs.shape[-1] == len(self.list_models)
        list_pred = []
        for _probs, model in zip(probs.T, self.list_models):
            list_pred.append(model.predict(_probs))
        output = npe.stack(list_pred, dtype=np.float64).T
        if is_binary:
            output = output[:, 0]
        LOGGER.info("END")
        return output


class TemperatureScaling(BaseCalibrator):
    def __init__(self, T: int | float=1.0):
        LOGGER.info("START")
        assert isinstance(T, (int, float))
        assert T > 0.0
        self.T = float(T)
        super().__init__()
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(T={self.T})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self) -> dict:
        return {
            "__BaseModel__": "kkmlmanager.calibration.TemperatureScaling",
            "T": self.T
        }
    @classmethod
    def from_dict(cls, dict_model: dict):
        assert isinstance(dict_model, dict)
        return cls(T=dict_model["T"])
    def fit(self, probs: np.ndarray, labels: np.ndarray, axis: int=-1):
        LOGGER.info("START")
        assert isinstance(probs,  np.ndarray)
        assert isinstance(labels, np.ndarray)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        if probs.ndim == 2 and probs.shape[1] == 1:
            probs = np.concatenate([1.0 - probs, probs], axis=-1)
        assert probs. ndim >= 2
        assert labels.ndim == 1
        assert probs.shape[0] == labels.shape[0]
        assert isinstance(axis, int)
        logits = np.log(probs + 1e-12)
        res = minimize(
            fun=lambda x: self.nll_and_grad(x[0], logits=logits, labels=labels, axis=axis),
            x0=np.array([self.T], dtype=np.float64),
            jac=True,
            bounds=[(1e-2, None)],
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        LOGGER.info(f"T: init={self.T}, after={float(res.x[0])}")
        self.T = float(res.x[0])
        LOGGER.info("END")
        return self
    def predict(self, probs: np.ndarray, axis: int=-1):
        LOGGER.info("START")
        assert isinstance(probs, np.ndarray)
        is_binary, is_single = False, False
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
            is_single = True
        if probs.ndim == 2 and probs.shape[1] == 1:
            probs = np.concatenate([1.0 - probs, probs], axis=-1)
            is_binary = True
        assert probs.ndim >= 2
        logits  = np.log(probs + 1e-12)
        calibrated = self.temperature_scaling(logits, self.T, axis)
        if is_binary:
            calibrated = calibrated[:, -1:]
        if is_single:
            calibrated = calibrated[:, -1]
        LOGGER.info("END")
        return calibrated
    @classmethod
    def temperature_scaling(cls, logits: np.ndarray, temperature: int | float=1.0, axis: int=-1):
        assert isinstance(logits, np.ndarray)
        assert logits.ndim >= 2
        assert isinstance(temperature, (int, float)) and temperature > 0.0
        assert isinstance(axis, int)
        centered = logits - np.max(logits, axis=axis, keepdims=True)
        scaled   = centered / temperature
        return softmax(scaled, axis=axis)
    @classmethod
    def nll_and_grad(cls, temperature: float | int, logits: np.ndarray=None, labels: np.ndarray=None, axis: int=-1):
        assert isinstance(temperature, (int, float)) and temperature > 0.0
        assert isinstance(logits, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert logits.ndim >= 2
        assert labels.ndim == 1
        assert logits.shape[0] == labels.shape[0]
        assert isinstance(axis, int)
        T        = float(temperature)
        centered = logits - np.max(logits, axis=axis, keepdims=True)
        scaled   = centered / temperature
        P        = softmax(scaled, axis=axis)
        idx      = np.arange(len(labels))
        nll      = -np.sum(np.log(P[idx, labels]))
        E_z      = np.sum(P * centered, axis=1)
        z_true   = centered[idx, labels]
        grad     = -np.sum((E_z - z_true) / (T**2))
        return nll, np.array([grad])


def calib_with_error(prob: np.ndarray, target: np.ndarray, n_bins: int=10):
    assert isinstance(prob, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert prob.shape == target.shape
    assert prob.ndim == target.ndim == 1
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


def calibration_curve_plot(
    prob_pre: np.ndarray, prob_aft: np.ndarray, target: np.ndarray, n_bins: int=10, figsize: tuple=(12, 8)
) -> dict[str, plt.figure]:
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


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 100, npow: float | int=1.0) -> float:
    assert isinstance(probs,  np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert probs.shape[0] == labels.shape[0]
    assert probs.ndim in [1, 2]
    assert labels.ndim == 1
    assert probs.dtype  in [np.float16, np.float32, np.float64, np.float128, float]
    assert labels.dtype in [np.int8, np.int16, np.int32, np.int64, int, bool, np.bool_]
    assert isinstance(n_bins, int) and n_bins >= 1
    assert isinstance(npow, (float, int)) and npow >= 0.1
    if probs.ndim > 1:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)
    N = labels.shape[0]
    bin_edges = (np.linspace(0.0, 1.0, n_bins + 1) ** npow)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        bin_size = np.sum(mask)
        if bin_size == 0:
            continue
        avg_conf = np.mean(confidences[mask])
        acc = np.mean(predictions[mask] == labels[mask])
        ece += (bin_size / N) * np.abs(acc - avg_conf)
    return ece
