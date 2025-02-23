import json, pickle, base64
from functools import partial
import numpy as np

# local package
from kkgbdt import KkGBDT
from kklogger import set_logger
from .util.com import check_type_list


__all__ = [
    "BaseModel",
    "MultiModel",
]


LOGGER = set_logger(__name__)
PICKLE_PROTOCOL = 5


class BaseModel:
    def __init__(self, func_predict: str, **kwargs):
        assert isinstance(func_predict, str)
        assert not hasattr(self, func_predict)
        self.func_predict = func_predict
        setattr(self, self.func_predict, partial(self.predict_common, **kwargs))
    def __str__(self):
        raise NotImplementedError()
    def to_dict(self) -> dict:
        raise NotImplementedError()
    def to_json(self) -> str:
        raise NotImplementedError()
    @classmethod
    def from_dict(cls, dict_model: dict):
        raise NotImplementedError()
    @classmethod
    def load_from_json(cls, json_model: str):
        raise NotImplementedError()
    def predict_common(self):
        raise NotImplementedError()


class MultiModel(BaseModel):
    def __init__(self, models: list[object], func_predict: str):
        LOGGER.info("START")
        assert isinstance(models, list) and len(models) > 0
        for x in models: assert hasattr(x, func_predict)
        assert isinstance(func_predict, str)
        classes_ = [[int(y) for y in x.classes_] if hasattr(x, "classes_") else None for x in models]
        for x in classes_: assert x == classes_[0]
        self.classes_ = classes_[0]
        self.models   = models
        super().__init__(func_predict)
        LOGGER.info("END")
    def __str__(self):
        return f"{self.__class__.__name__}(models={self.models})"
    def __repr__(self):
        return self.__str__()
    def to_dict(self) -> dict:
        return {
            "classes_": self.classes_.copy(),
            "func_predict": self.func_predict,
            "models": [x.to_dict() if isinstance(x, KkGBDT) else base64.b64encode(pickle.dumps(x, protocol=PICKLE_PROTOCOL)).decode('ascii') for x in self.models],
        }
    def to_json(self, indent: int=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    @classmethod
    def from_dict(cls, dict_model: dict):
        assert isinstance(dict_model, dict)
        func_p = dict_model["func_predict"]
        models = [pickle.loads(base64.b64decode(x)) if isinstance(x, str) else KkGBDT.from_dict(x) for x in dict_model["models"]]
        return cls(models, func_p)
    @classmethod
    def load_from_json(cls, json_model: str):
        assert isinstance(json_model, str)
        return cls.from_dict(json.loads(json_model))
    def predict_common(self, input: np.ndarray, weight: list[float]=None, **kwargs) -> np.ndarray:
        LOGGER.info(f"START {self.__class__}")
        LOGGER.info(f"model predict function: '{self.func_predict}'")
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
