import json, importlib
from functools import partial
import numpy as np
from kklogger import set_logger

# local package
from .calibration import MultiLabelRegressionWithError, calibration_curve_plot, TemperatureScaling, BaseCalibrator, expected_calibration_error
from .util.com import check_type_list, encode_object, decode_object
from .util.numpy import NdarrayWithErr, nperr_stack
LOGGER = set_logger(__name__)


__all__ = [
    "BaseModel",
    "MultiModel",
    "Calibrator",
]


class BaseModel:
    def __init__(self, func_predict: str, default_params_for_predict: dict={}):
        assert isinstance(func_predict, str)
        assert not hasattr(self, func_predict)
        assert isinstance(default_params_for_predict, dict)
        self.func_predict = func_predict
        setattr(self, self.func_predict, partial(self._predict_common, **default_params_for_predict))
    def __str__(self):
        raise NotImplementedError()
    def to_dict(self) -> dict:
        raise NotImplementedError()
    def to_json(self, indent: int=None, mode: int=0, savedir: str=None):
        return json.dumps(self.to_dict(mode=mode, savedir=savedir), indent=indent)
    @classmethod
    def from_dict(cls, dict_model: dict):
        raise NotImplementedError()
    @classmethod
    def from_json(cls, json_model: str, basedir: str=None):
        return cls.from_dict(json.loads(json_model), basedir=basedir)
    def _predict_common(self):
        raise NotImplementedError()
    def dump_with_loader(self):
        raise NotImplementedError()


class MultiModel(BaseModel):
    def __init__(self, models: list[object], func_predict: str, default_params_for_predict: dict={}):
        LOGGER.info("START")
        assert isinstance(models, list) and len(models) > 0
        for x in models: assert hasattr(x, func_predict)
        assert isinstance(func_predict, str)
        classes_ = [[int(y) for y in x.classes_] if hasattr(x, "classes_") else None for x in models]
        for x in classes_: assert x == classes_[0]
        self.classes_ = classes_[0]
        self.models   = models
        super().__init__(func_predict, default_params_for_predict=default_params_for_predict)
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(models={self.models}, func_predict={self.func_predict}, {self.func_predict}={getattr(self, self.func_predict)})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self, mode: int=0, savedir: str=None) -> dict:
        assert isinstance(mode, int) and mode in [0, 1, 2]
        models = []
        for _model in self.models:
            if isinstance(_model, BaseModel):
                models.append(_model.to_dict(mode=mode, savedir=savedir))
            else:
                models.append(encode_object(_model, mode=mode, savedir=savedir))
        return {
            "__BaseModel__": "kkmlmanager.models.MultiModel",
            "classes_": self.classes_.copy() if self.classes_ is not None else None,
            "func_predict": self.func_predict,
            "models": models,
        }
    @classmethod
    def from_dict(cls, dict_model: dict, basedir: str=None):
        assert isinstance(dict_model, dict)
        models = []
        for x in dict_model["models"]:
            if "__BaseModel__" in x:
                _path, _cls = x["__BaseModel__"].rsplit(".", 1)
                _cls = getattr(importlib.import_module(_path), _cls)
                models.append(_cls.from_dict(x, basedir=basedir))
            else:
                models.append(decode_object(x, basedir=basedir))
        return cls(models, dict_model["func_predict"])
    def _predict_common(self, input: np.ndarray, weight: list[float]=None, **kwargs) -> np.ndarray:
        LOGGER.info(f"START {self.__class__}")
        LOGGER.info(f"model predict function: '{self.func_predict}', weight: {weight}, kwargs: {kwargs}")
        assert isinstance(input, np.ndarray)
        assert len(input.shape) >= 2
        if weight is None: weight = [1.0] * len(self.models)
        assert check_type_list(weight, (int, float))
        output = None
        for i, model in enumerate(self.models):
            LOGGER.info(f"predict model [{i}], model: {model}")
            _output = getattr(model, self.func_predict)(input, **kwargs) * weight[i]
            if output is None: output  = _output
            else:              output += _output
        output = output / sum(weight)
        LOGGER.info("END")
        return output
    def dump_with_loader(self):
        return {
            "__class__": "kkmlmanager.models.MultiModel",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(mode=0, savedir=None),
        }


class Calibrator(BaseModel):
    def __init__(self, model, func_predict: str, is_normalize: bool=False, is_reg: bool=False, is_binary_fit: bool=False):
        LOGGER.info("START")
        assert isinstance(func_predict, str)
        assert hasattr(model, func_predict)
        assert isinstance(is_normalize, bool)
        assert isinstance(is_reg, bool)
        assert isinstance(is_binary_fit, bool)
        if is_reg:
            assert is_binary_fit == False
            assert is_normalize  == False
        LOGGER.info(f"model: {model}, func_predict: {func_predict}, is_normalize: {is_normalize}, is_reg: {is_reg}, is_binary_fit: {is_binary_fit}")
        self.model         = model
        self.calibrator    = MultiLabelRegressionWithError(increasing=True, set_first_score=True)
        self.func_predict  = func_predict
        self.is_normalize  = is_normalize
        self.is_reg        = is_reg
        self.is_binary_fit = is_binary_fit
        self.classes_      = None
        self.calib_fig_wo_norm    = None
        self.calib_fig_with_norm1 = None
        self.calib_fig_with_norm2 = None
        super().__init__(func_predict, default_params_for_predict={"is_mock": False})
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model}, calibrator={self.calibrator}, func_predict={self.func_predict}, is_normalize={self.is_normalize}, is_reg={self.is_reg}, is_binary_fit={self.is_binary_fit})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self, mode: int=0, savedir: str=None) -> dict:
        assert isinstance(mode, int) and mode in [0, 1, 2]
        if isinstance(self.model, BaseModel):
            model = self.model.to_dict(mode=mode, savedir=savedir)
        else:
            model = encode_object(self.model, mode=mode, savedir=savedir)
        return {
            "__BaseModel__": "kkmlmanager.models.Calibrator",
            "model": model,
            "calibrator": self.calibrator.to_dict(),
            "func_predict": self.func_predict,
            "is_normalize": self.is_normalize,
            "is_reg": self.is_reg,
            "is_binary_fit": self.is_binary_fit,
            "classes_": self.classes_.tolist() if self.classes_ is not None else None,
            "calib_fig_wo_norm":    {
                x: encode_object(y, mode={0:0,1:0,2:2}[mode]) for x, y in self.calib_fig_wo_norm.items()
            } if self.calib_fig_wo_norm    is not None else None,
            "calib_fig_with_norm1": {
                x: encode_object(y, mode={0:0,1:0,2:2}[mode]) for x, y in self.calib_fig_with_norm1.items()
            } if self.calib_fig_with_norm1 is not None else None,
            "calib_fig_with_norm2": {
                x: encode_object(y, mode={0:0,1:0,2:2}[mode]) for x, y in self.calib_fig_with_norm2.items()
            } if self.calib_fig_with_norm2 is not None else None,
        }
    @classmethod
    def from_dict(cls, dict_model: dict, basedir: str=None):
        assert isinstance(dict_model, dict)
        model = dict_model["model"]
        if "__BaseModel__" in model:
            _path, _cls = model["__BaseModel__"].rsplit(".", 1)
            _cls  = getattr(importlib.import_module(_path), _cls)
            model = _cls.from_dict(model, basedir=basedir)
        else:
            model = decode_object(model, basedir=basedir)
        ins            = cls(model, dict_model["func_predict"], dict_model["is_normalize"], dict_model["is_reg"], dict_model["is_binary_fit"])
        ins.calibrator = MultiLabelRegressionWithError.from_dict(dict_model["calibrator"])
        ins.classes_   = np.array(dict_model["classes_"]) if dict_model["classes_"] is not None else None
        ins.calib_fig_wo_norm    = {
            x: decode_object(y) for x, y in dict_model["calib_fig_wo_norm"].items()
        } if dict_model["calib_fig_wo_norm"]    is not None else None
        ins.calib_fig_with_norm1 = {
            x: decode_object(y) for x, y in dict_model["calib_fig_with_norm1"].items()
        } if dict_model["calib_fig_with_norm1"] is not None else None
        ins.calib_fig_with_norm2 = {
            x: decode_object(y) for x, y in dict_model["calib_fig_with_norm2"].items()
        } if dict_model["calib_fig_with_norm2"] is not None else None
        return ins
    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, is_input_prob: bool=False, n_bootstrap: int=100, n_bins: int=20, **kwargs):
        """
        if is_input_prob == True, 'input_x' must be probabilities, not Features.
        """
        LOGGER.info("START")
        LOGGER.info(f"input_x.shape: {input_x.shape}, input_y.shape: {input_y.shape}, is_input_prob: {is_input_prob}, n_bootstrap: {n_bootstrap}, n_bins: {n_bins}")
        assert isinstance(input_x, np.ndarray) and (len(input_x.shape) == 2 and input_x.shape[-1] >= 1)
        assert isinstance(input_y, np.ndarray) and len(input_y.shape) == 1
        assert isinstance(is_input_prob, bool)
        assert isinstance(n_bootstrap, int) and n_bootstrap >= 1
        assert isinstance(n_bins, int) and n_bins >= 5
        if self.is_reg:
            self.classes_ = np.arange(1, dtype=int) if len(input_y.shape) == 1 else np.arange(input_y.shape[-1], dtype=int)
        else:
            self.classes_ = np.sort(np.unique(input_y))
        if is_input_prob == False:
            input_x = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
        if self.is_binary_fit:
            LOGGER.info("set for binary fitting.")
            assert input_x.shape[-1] >= 3
            assert np.all(self.classes_ == np.arange(input_x.shape[-1], dtype=self.classes_.dtype))
            assert self.is_reg == False
            input_y = np.eye(input_x.shape[-1])[input_y].reshape(-1).astype(int)
            input_x = input_x.reshape(-1, 1)
            self.classes_ = np.array([0, 1])
        self.calibrator.fit(input_x, input_y, is_reg=self.is_reg, n_bootstrap=n_bootstrap)
        output = self._predict_common(input_x, is_mock=True)
        if len(output.shape) == 1:
            output = nperr_stack([1 - output, output], dtype=np.float64).T
        if not self.is_reg:
            ndfwk = output.to_numpy()
            self.calib_fig_wo_norm    = calibration_curve_plot(input_x, ndfwk,                                     input_y, n_bins=n_bins)
            self.calib_fig_with_norm1 = calibration_curve_plot(input_x, ndfwk / ndfwk.sum(axis=-1, keepdims=True), input_y, n_bins=n_bins)
            self.calib_fig_with_norm2 = calibration_curve_plot(input_x, output.to_numpy(axis_normalize=-1),        input_y, n_bins=n_bins)
        LOGGER.info("END")
        return self
    def _predict_common(self, input_x, *args, is_mock: bool=False, is_normalize: bool=None, **kwargs) -> NdarrayWithErr:
        """
        Note::
            If is_mock == False, "input_x" must be features.
            If is_mock == True,  "input_x" must be probabilities.
        """
        LOGGER.info(f"START {self.__class__}")
        assert isinstance(is_mock, bool)
        assert isinstance(is_normalize, (bool, type(None)))
        is_normalize = self.is_normalize if is_normalize is None else is_normalize
        LOGGER.info(f"is_mock: {is_mock}, model predict function: '{self.func_predict}', is_normalize: {is_normalize}, is_binary_fit: {self.is_binary_fit}")
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
            LOGGER.info("binary fitting mode.")
            output = output.reshape(-1, 1)
        LOGGER.info("calibrate output ...")
        output = self.calibrator.predict(output)
        if self.is_binary_fit:
            output = output[:, -1]
            output = output.reshape(-1, classes)
        if is_normalize:
            LOGGER.info("normalize output ...")
            if len(output.shape) == 2 and output.shape[-1] >= 2:
                output = output.to_numpy(axis_normalize=-1)
            else:
                LOGGER.warning(f"This shape ({output.shape}) is not supported for normalization.")
        if is_binary:
            # return to same shape as model's output
            output = output[:, -1]
        LOGGER.info("END")
        return output
    def dump_with_loader(self):
        return {
            "__class__": "kkmlmanager.models.Calibrator",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(mode=0, savedir=None),
        }
