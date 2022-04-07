import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

# local package
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "predict_model",
    "eval_model",
]


def accuracy_top_k(answer: np.ndarray, input: np.ndarray, top_k: int=1):
    assert isinstance(input,  np.ndarray)
    assert isinstance(answer, np.ndarray)
    assert len(answer.shape) == 1
    assert (input.shape[0] == answer.shape[0])
    assert isinstance(top_k, int) and top_k >= 1
    ndf = np.argsort(input, axis=1)[:, ::-1]
    ndf = ndf[:, :top_k]
    return (ndf == answer.reshape(-1, 1)).sum() / answer.shape[0]

def predict_model(model, input: np.ndarray):
    logger.info("START")
    assert isinstance(input, np.ndarray)
    assert len(input.shape) == 2
    df = pd.DataFrame(index=np.arange(input.shape[0]))
    if hasattr(model, "predict"):
        if hasattr(model, "predict_proba"):
            output = model.predict_proba(input)
            assert isinstance(output, np.ndarray)
            assert len(output.shape) == 2
            ndf_class = model.classes_ if hasattr(model, "classes_") else np.arange(output.shape[1])
            df[[f"predict_proba_{i}" for i in ndf_class.astype(int)]] = output
            df["predict"] = np.argmax(output, axis=1)
        else:
            output = model.predict(input)
            assert isinstance(output, np.ndarray)
            assert len(output.shape) == 1
            df["predict"] = output
    logger.info("END")
    return df

def eval_model(model, input_x: np.ndarray, input_y: np.ndarray, is_reg: bool=False):
    logger.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    df = predict_model(model, input_x)
    se = pd.Series(dtype=object)
    if is_reg:
        assert len(input_y.shape) == 1
        df["answer"] = input_y
        se["r2"]   = r2_score(input_y, df["predict"])
        se["rmse"] = np.sqrt( ((input_y - df["predict"].values) ** 2).sum() / input_y.shape[0] )
        se["mae"]  = (np.abs(input_y - df["predict"].values)).sum() / input_y.shape[0]
    else:
        assert len(input_y.shape) in [1,2]
        if len(input_y.shape) == 1:
            df["answer"] = input_y
        else:
            df["answer"] = np.argmax(input_y, axis=1)
            df[[f"answer_{i}" for i in range(input_y.shape[1])]] = input_y
        n_class    = df.columns.str.contains("^predict_proba_", regex=True).sum()
        ndf_class  = model.classes_ if hasattr(model, "classes_") else np.arange(n_class)
        dict_class = {x:i for i, x in enumerate(ndf_class)}
        ndf_pred   = df[[f"predict_proba_{i_class}" for i_class in ndf_class]].values
        if len(input_y.shape) == 1:
            input_y = np.vectorize(lambda x: dict_class.get(x))(input_y.copy()).astype(int)
            input_y = np.eye(n_class)[input_y]
        assert ndf_pred.shape == input_y.shape
        ndf_pred       = np.clip(ndf_pred, 1e-10, 1)
        input_y_class  = np.argmax(input_y, axis=1)
        input_y_argmax = np.zeros_like(input_y)
        input_y_argmax[np.arange(input_y_argmax.shape[0]), input_y_class] = 1
        se["logloss"]        = (-1 * input_y        * np.log(ndf_pred)).sum(axis=1).mean()
        se["logloss_argmax"] = (-1 * input_y_argmax * np.log(ndf_pred)).sum(axis=1).mean()
        for i in range(1, n_class+1):
            strlen=len(str(n_class))
            se[f"acc_top{str(i).zfill(strlen)}"] = accuracy_top_k(input_y_class, ndf_pred, top_k=i)
        for i, i_class in dict_class.items():
            se[f"auc_{i_class}"] = roc_auc_score(input_y[:, i], ndf_pred[:, i])
        weight = np.bincount(input_y_class, minlength=n_class)
        se[f"auc_multi"] = (se.loc[se.index.str.contains("^auc_")].values * weight).sum() / weight.sum()
    logger.info("END")
    return se, df
