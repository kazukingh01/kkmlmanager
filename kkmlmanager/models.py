import json, importlib
from functools import partial
import numpy as np
import pandas as pd
import polars as pl
from kklogger import set_logger

# local package
from .calibration import MultiLabelRegressionWithError, calibration_curve_plot, \
    TemperatureScaling, BaseCalibrator
from .procs import mask_values_for_json
from .util.com import check_type_list, encode_object, decode_object
from .util.numpy import NdarrayWithErr
LOGGER = set_logger(__name__)


__all__ = [
    "BaseModel",
    "MultiModel",
    "Calibrator",
    "ChainModel",
]


class BaseModel:
    def __init__(self, func_predict: str, default_params_for_predict: dict={}):
        assert isinstance(func_predict, str)
        assert not hasattr(self, func_predict)
        assert isinstance(default_params_for_predict, dict)
        self.func_predict = func_predict
        self.default_params_for_predict = default_params_for_predict
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
        LOGGER.info(f"START: {self.__class__.__name__}")
        assert isinstance(models, list) and len(models) > 0
        for x in models: assert hasattr(x, func_predict)
        assert isinstance(func_predict, str)
        classes_ = [[int(y) for y in x.classes_] if hasattr(x, "classes_") else None for x in models]
        for x in classes_: assert x == classes_[0]
        self.classes_ = classes_[0]
        self.models   = models
        super().__init__(func_predict, default_params_for_predict=default_params_for_predict)
        LOGGER.info(f"END: {self.__class__.__name__}")
    def __str__(self):
        return f"{self.__class__.__name__}(models={self.models}, func_predict={self.func_predict}, default_params_for_predict={self.default_params_for_predict})"
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
            "default_params_for_predict": {k: mask_values_for_json(v) for k, v in self.default_params_for_predict.items()},
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
        return cls(models, dict_model["func_predict"], default_params_for_predict=dict_model["default_params_for_predict"])
    def _predict_common(self, input: np.ndarray, weight: list[float]=None, **kwargs) -> np.ndarray:
        LOGGER.info(f"START: {self.__class__.__name__}")
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
        LOGGER.info(f"END: {self.__class__.__name__}")
        return output
    def dump_with_loader(self):
        return {
            "__class__": "kkmlmanager.models.MultiModel",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(mode=0, savedir=None),
        }


class Calibrator(BaseModel):
    def __init__(self, model, func_predict: str, is_normalize: bool=False, is_reg: bool=False, calibmodel: str | int=None):
        LOGGER.info(f"START: {self.__class__.__name__}")
        assert isinstance(func_predict, str)
        assert hasattr(model, func_predict)
        assert isinstance(is_normalize, bool)
        assert isinstance(is_reg, bool)
        assert isinstance(calibmodel, (str, int))
        if isinstance(calibmodel, int):
            assert calibmodel in [0, 1]
            calibmodel = ["MultiLabelRegressionWithError", "TemperatureScaling"][calibmodel]
        assert calibmodel in ["MultiLabelRegressionWithError", "TemperatureScaling"]
        if is_reg:
            assert is_normalize == False
            assert calibmodel   in ["MultiLabelRegressionWithError"]
        LOGGER.info(f"model: {model}, func_predict: {func_predict}, is_normalize: {is_normalize}, is_reg: {is_reg}, calibmodel: {calibmodel}")
        self.model         = model
        self.calibrator: BaseCalibrator = {
            "MultiLabelRegressionWithError": MultiLabelRegressionWithError(increasing=True, set_first_score=True, is_reg=is_reg),
            "TemperatureScaling":            TemperatureScaling(),
        }[calibmodel]
        self.func_predict  = func_predict
        self.is_normalize  = is_normalize
        self.is_reg        = is_reg
        self.calibmodel    = calibmodel
        self.classes_      = None
        self.calib_fig_wo_norm    = None
        self.calib_fig_with_norm1 = None
        self.calib_fig_with_norm2 = None
        super().__init__(func_predict, default_params_for_predict={"is_mock": False})
        LOGGER.info(f"END: {self.__class__.__name__}")
    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model}, calibrator={self.calibrator}, func_predict={self.func_predict}, is_normalize={self.is_normalize}, is_reg={self.is_reg}, calibmodel={self.calibmodel})"
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
            "calibmodel": self.calibmodel,
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
        ins            = cls(model, dict_model["func_predict"], dict_model["is_normalize"], dict_model["is_reg"], calibmodel=dict_model["calibmodel"])
        _path, _cls    = dict_model["calibrator"]["__BaseModel__"].rsplit(".", 1)
        _cls: BaseCalibrator = getattr(importlib.import_module(_path), _cls)
        ins.calibrator = _cls.from_dict(dict_model["calibrator"])
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
    def fit(self, input_x: np.ndarray, input_y: np.ndarray, *args, is_input_prob: bool=False, n_bins: int=20, **kwargs):
        """
        if is_input_prob == True, 'input_x' must be probabilities, not Features.
        """
        LOGGER.info(f"START: {self.__class__.__name__}")
        assert isinstance(input_y, np.ndarray) and input_y.ndim in [1, 2]
        assert isinstance(is_input_prob, bool)
        assert isinstance(n_bins, int) and n_bins >= 5
        if is_input_prob == False:
            input_x = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
        LOGGER.info(f"input_x.shape: {input_x.shape}, input_y.shape: {input_y.shape}, is_input_prob: {is_input_prob}, n_bins: {n_bins}")
        assert isinstance(input_x, np.ndarray) and input_x.ndim in [1, 2]          
        if self.is_reg:
            self.classes_ = np.arange(1, dtype=int) if input_y.ndim == 1 else np.arange(input_y.shape[-1], dtype=int)
            assert input_x.shape == input_y.shape
        else:
            self.classes_ = np.sort(np.unique(input_y)).astype(int)
            assert self.classes_.shape[0] >= 2
            assert input_x.shape[0] == input_y.shape[0]
            if input_x.ndim == 1:
                assert np.allclose(self.classes_, np.array([0, 1]))
                input_x = input_x.reshape(-1, 1)
                input_x = np.concatenate([1 - input_x, input_x], axis=-1)
            elif input_x.shape[1] == 1 and self.classes_.shape[0] == 2:
                assert np.allclose(self.classes_, np.array([0, 1]))
                input_x = np.concatenate([1 - input_x, input_x], axis=-1)
        self.calibrator.fit(input_x, input_y)
        output = self._predict_common(input_x, is_mock=True)
        assert isinstance(output, (np.ndarray, NdarrayWithErr))
        assert output.ndim == 2
        if not self.is_reg:
            if isinstance(output, NdarrayWithErr):
                ndfwk = output.to_numpy()
            else:
                ndfwk = output
            self.calib_fig_wo_norm      = calibration_curve_plot(input_x, ndfwk,                                     input_y, n_bins=n_bins)
            # self.calib_fig_with_norm1 = calibration_curve_plot(input_x, ndfwk / ndfwk.sum(axis=-1, keepdims=True), input_y, n_bins=n_bins)
            # self.calib_fig_with_norm2 = calibration_curve_plot(input_x, output.to_numpy(axis_normalize=-1),        input_y, n_bins=n_bins)
        LOGGER.info(f"END: {self.__class__.__name__}")
        return self
    def _predict_common(self, input_x, *args, is_mock: bool=False, is_normalize: bool=None, **kwargs) -> NdarrayWithErr:
        """
        Note::
            If is_mock == False, "input_x" must be features.
            If is_mock == True,  "input_x" must be probabilities.
        """
        LOGGER.info(f"START: {self.__class__.__name__}")
        assert isinstance(is_mock, bool)
        assert isinstance(is_normalize, (bool, type(None)))
        is_normalize = self.is_normalize if is_normalize is None else is_normalize
        LOGGER.info(f"is_mock: {is_mock}, model predict function: '{self.func_predict}', is_normalize: {is_normalize}")
        is_binary = False
        if is_mock:
            output = input_x
        else:
            output = getattr(self.model, self.func_predict)(input_x, *args, **kwargs)
            if len(output.shape) == 1:
                output    = output.reshape(-1, 1)
                is_binary = True
        LOGGER.info("calibrate output ...")
        output = self.calibrator.predict(output)
        if is_normalize:
            LOGGER.info("normalize output ...")
            if len(output.shape) == 2 and output.shape[-1] >= 2:
                if isinstance(output, NdarrayWithErr):
                    output = output.to_numpy()
                output = output / output.sum(axis=-1, keepdims=True)
            else:
                LOGGER.warning(f"This shape ({output.shape}) is not supported for normalization.")
        if is_binary:
            # return to same shape as model's output
            output = output[:, -1]
        LOGGER.info(f"END: {self.__class__.__name__}")
        return output
    def dump_with_loader(self):
        return {
            "__class__": "kkmlmanager.models.Calibrator",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(mode=0, savedir=None),
        }


class ChainModel(BaseModel):
    def __init__(self, models: list[dict], eval_pre: str, eval_post: str, func_predict: str, default_params_for_predict: dict={}):
        LOGGER.info(f"START: {self.__class__.__name__}")
        assert isinstance(models, list) and len(models) > 0
        assert check_type_list(models, dict)
        listwk = []
        for x in models:
            assert "model" in x
            assert "name"  in x and isinstance(x["name"], str)
            assert "eval"  in x and isinstance(x["eval"], str)
            assert "shape" in x and isinstance(x["shape"], tuple) and check_type_list(x["shape"], int)
            for y in x["shape"]: assert y == -1 or y > 0
            assert x["name"] not in listwk
            listwk.append(x["name"])
        assert isinstance(eval_pre, str)
        assert isinstance(eval_post, str)
        assert isinstance(func_predict, str)
        self.models    = models
        self.eval_pre  = eval_pre
        self.eval_post = eval_post
        super().__init__(func_predict, default_params_for_predict=default_params_for_predict)
        LOGGER.info(f"END: {self.__class__.__name__}")
    def __str__(self):
        return (
            f"{self.__class__.__name__}(models={self.models}, eval_pre={self.eval_pre}, eval_post={self.eval_post}, " + 
            f"func_predict={self.func_predict}, default_params_for_predict={self.default_params_for_predict})"
        )
    def __repr__(self):
        return self.__str__()
    def to_dict(self, mode: int=0, savedir: str=None) -> dict:
        assert isinstance(mode, int) and mode in [0, 1, 2]
        models = []
        for dictwk in self.models:
            _model = dictwk["model"]
            if isinstance(_model, BaseModel) or _model.__class__.__name__ == "MLManager":
                model = _model.to_dict(mode=mode, savedir=savedir)
            else:
                model = encode_object(_model, mode=mode, savedir=savedir)
            models.append({"name": dictwk["name"], "model": model, "eval": dictwk["eval"], "shape": list(dictwk["shape"])})
        return {
            "__BaseModel__": "kkmlmanager.models.ChainModel",
            "eval_pre":     self.eval_pre,
            "eval_post":    self.eval_post,
            "func_predict": self.func_predict,
            "default_params_for_predict": {k: mask_values_for_json(v) for k, v in self.default_params_for_predict.items()},
            "models":       models,
        }
    @classmethod
    def from_dict(cls, dict_model: dict, basedir: str=None):
        assert isinstance(dict_model, dict)
        models = []
        for x in dict_model["models"]:
            if "__BaseModel__" in x["model"]:
                _path, _cls = x["model"]["__BaseModel__"].rsplit(".", 1)
                _cls        = getattr(importlib.import_module(_path), _cls)
                model       = _cls.from_dict(x["model"], basedir=basedir)
            else:
                model       = decode_object(x["model"], basedir=basedir)
            models.append({"name": x["name"], "model": model, "eval": x["eval"], "shape": tuple(x["shape"])})
        return cls(
            models, dict_model["eval_pre"], dict_model["eval_post"], dict_model["func_predict"],
            default_params_for_predict=dict_model["default_params_for_predict"]
        )
    def check_eval(self, input_x: np.ndarray, **kwargs):
        ndfs     = {x["name"]: np.random.rand(*[(y if y > 0 else 10) for y in x["shape"]]) for x in self.models}
        dict_all = {"input": input_x, "input_pre": input_x, "models": self.models, "np": np, "pl": pl, "pd": pd} | kwargs | ndfs
        output   = eval(self.eval_post, {}, dict_all)
        LOGGER.info(f"type: {type(output)}, output: {output}")
    def _predict_common(self, input_x: np.ndarray, **kwargs):
        """
        You can use key below.
        - input: input data
        - input_pre: output of self.eval_pre
        - model: each model
        - models: self.models
        - np: numpy
        - pl: polars
        - pd: pandas
        - **names: ouptuts via self.model[i]["eval]
        - **kwargs
        """
        LOGGER.info(f"START: {self.__class__.__name__}")
        dict_all = {"input": input_x, "models": self.models, "np": np, "pl": pl, "pd": pd} | kwargs
        dict_all = dict_all | {"input_pre": eval(self.eval_pre, {}, dict_all)}
        for i, dictwk in enumerate(self.models):
            LOGGER.info(f"predict model [{i}], name: {dictwk['name']}, model: {dictwk['model']}")
            output   = eval(dictwk["eval"], {}, dict_all | {"model": dictwk["model"]})
            dict_all = dict_all | {dictwk["name"]: output}
        output = eval(self.eval_post, {}, dict_all)
        LOGGER.info(f"END: {self.__class__.__name__}")
        return output
    def dump_with_loader(self):
        return {
            "__class__": "kkmlmanager.models.ChainModel",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(mode=0, savedir=None),
        }
        
