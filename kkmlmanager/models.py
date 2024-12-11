import json
from functools import partial
import numpy as np

# local package
from kkmlmanager.util.com import check_type_list
from kklogger import set_logger
logger = set_logger(__name__)


__all__ = [
    "BaseModel",
    "MultiModel",
]


class BaseModel:
    def __init__(self, func_predict: str, **kwargs):
        assert isinstance(func_predict, str)
        self.func_predict = func_predict
        assert not hasattr(self, func_predict)
        setattr(self, self.func_predict, partial(self.predict_common, **kwargs))
    
    def to_json(self) -> dict:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def predict_common(self):
        raise NotImplementedError()


class MultiModel(BaseModel):
    def __init__(self, models: list[object], func_predict: str):
        logger.info("START")
        assert isinstance(models, list) and len(models) > 0
        assert isinstance(func_predict, str)
        self.classes_ = models[0].classes_ if hasattr(models[0], "classes_") else None
        for model in models:
            # Each model must have the function whose name is showed by "func_predict"
            assert hasattr(model, func_predict)
            if self.classes_ is not None:
                assert np.all(self.classes_ == model.classes_)
        self.models = models
        super().__init__(func_predict)
        logger.info("END")
    
    def to_json(self) -> dict:
        return {
            "func_predict": self.func_predict,
            "classes": str(self.classes_),
            "models": [x.__class__.__name__ for x in self.models],
            f"{self.func_predict}": str(getattr(self, self.func_predict)),
        }
    
    def __str__(self):
        return self.__class__.__name__ + " " + json.dumps(self.to_json(), indent=4)

    def predict_common(self, input: np.ndarray, weight: list[float]=None, **kwargs) -> np.ndarray:
        logger.info(f"START {self.__class__}")
        logger.info(f"model predict function: '{self.func_predict}'")
        assert isinstance(input, np.ndarray)
        if weight is None: weight = [1.0] * len(self.models)
        assert check_type_list(weight, float)
        output = None
        for i, model in enumerate(self.models):
            logger.info(f"predict model {i}, model: {model}")
            _output = getattr(model, self.func_predict)(input, **kwargs) * weight[i]
            if output is None: output  = _output
            else:              output += _output
        output = output / sum(weight)
        logger.info("END")
        return output

