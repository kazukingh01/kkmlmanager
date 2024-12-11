import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, log_loss

# local package
from kklogger import set_logger
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

def predict_model(model, input: np.ndarray, is_reg: bool=False, func_predict: str=None, **kwargs):
    logger.info("START")
    assert isinstance(input, np.ndarray)
    assert len(input.shape) == 2
    df = pd.DataFrame(index=np.arange(input.shape[0]))
    if func_predict is None:
        if   hasattr(model, "predict_proba"): func_predict = "predict_proba"
        elif hasattr(model, "predict"):       func_predict = "predict"
    logger.info(f"model: {model}, is_reg: {is_reg}, func_predict: {func_predict}, kwargs: {kwargs}")
    assert isinstance(func_predict, str)    
    output = getattr(model, func_predict)(input, **kwargs)
    assert isinstance(output, np.ndarray)
    assert len(output.shape) in [1, 2]
    if len(output.shape) == 1: output = output.reshape(-1, 1)
    if is_reg:
        df[[f"predict_{i}" for i in range(output.shape[-1])]] = output
        logger.info(f"we define 'predict' as 'predict_0'.")
        df["predict"] = df["predict_0"].copy()
    else:
        if output.shape[-1] == 1:
            output = np.concatenate([(1 - output), output], axis=1)
        if hasattr(model, "classes_") and isinstance(model.classes_, np.ndarray):
            ndf_class = model.classes_ 
            if ndf_class.shape[0] != output.shape[-1]:
                logger.warning(f"shape is different. output: {output.shape}, model.classes_: {ndf_class.shape}")
                ndf_class = np.arange(output.shape[1])
        else:
            ndf_class = np.arange(output.shape[1])
        assert int(ndf_class.sum()) == int(np.arange(ndf_class.shape[0]).sum()) # allow [0, 1, 2, ...]. Don't allow [0, 2, 3, ...]
        df[[f"predict_proba_{i}" for i in ndf_class.astype(int)]] = output
        df["predict"] = np.argmax(output, axis=1)
    logger.info("END")
    return df

def eval_model(input_x: np.ndarray, input_y: np.ndarray, model=None, is_reg: bool=False, func_predict: str=None, **kwargs):
    logger.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    logger.info(f"model: {model}, is_reg: {is_reg}, func_predict: {func_predict}, kwargs: {kwargs}")
    if model is None:
        if is_reg:
            assert len(input_x.shape) == 1
            df = pd.DataFrame(input_x, columns=["predict"]) 
        else:
            if len(input_x.shape) == 1:
                input_x = np.concatenate([(1 - input_x).reshape(-1, 1), input_x.reshape(-1, 1)], axis=1)
            assert len(input_x.shape) == 2
            df = pd.DataFrame(input_x, columns=[f"predict_proba_{i}" for i in range(input_x.shape[1])]) 
    else:
        df = predict_model(model, input_x, is_reg=is_reg, func_predict=func_predict, **kwargs)
    se = pd.Series(dtype=object)
    se["n_data"] = df.shape[0]
    if is_reg:
        assert len(input_y.shape) == 1
        df["answer"] = input_y
        se["r2"]   = r2_score(input_y, df["predict"])
        se["rmse"] = np.sqrt( ((input_y - df["predict"].values) ** 2).sum() / input_y.shape[0] )
        se["mae"]  = (np.abs(input_y - df["predict"].values)).sum() / input_y.shape[0]
    else:
        assert len(input_y.shape) in [1,2]
        n_class  = df.columns.str.contains("^predict_proba_", regex=True).sum()
        ndf_pred = df[[f"predict_proba_{i_class}" for i_class in np.arange(n_class, dtype=int)]].values
        if len(input_y.shape) == 1:
            df["answer"] = input_y
        else:
            if ndf_pred.shape[-1] < input_y.shape[-1]:
                logger.warning(f"shape is different. output: {ndf_pred.shape}, answer: {input_y.shape}")
                input_y = input_y[:, :ndf_pred.shape[-1]]
            df["answer"] = np.argmax(input_y, axis=1)
            df[[f"answer_{i}" for i in range(input_y.shape[1])]] = input_y
        if len(input_y.shape) == 1:
            input_y = np.eye(n_class)[input_y.astype(int)]
        assert ndf_pred.shape == input_y.shape
        ndf_pred       = np.clip(ndf_pred, 1e-10, 1)
        input_y_class  = np.argmax(input_y, axis=1)
        input_y_argmax = np.zeros_like(input_y, dtype=int)
        input_y_argmax[np.arange(input_y_argmax.shape[0]), input_y_class] = 1
        se["logloss"]        = (-1 * input_y        * np.log(ndf_pred)).sum(axis=1).mean()
        se["logloss_argmax"] = (-1 * input_y_argmax * np.log(ndf_pred)).sum(axis=1).mean()
        for i in np.arange(n_class):
            try: tmp = log_loss(input_y_argmax[:, i], ndf_pred[:, i])
            except ValueError: tmp = float("nan")
            se[f"binary_logloss_{i}"] = tmp
        for i in range(1, n_class+1):
            strlen=len(str(n_class))
            se[f"acc_top{str(i).zfill(strlen)}"] = accuracy_top_k(input_y_class, ndf_pred, top_k=i)
        for i in np.arange(n_class):
            try: tmp = roc_auc_score(input_y_argmax[:, i], ndf_pred[:, i])
            except ValueError: tmp = float("nan")
            se[f"auc_{i}"] = tmp
        weight = np.bincount(input_y_class, minlength=n_class)
        se["auc_multi"] = (se.loc[se.index.str.contains("^auc_")].values * weight).sum() / weight.sum()
        se["auc_all"]   = roc_auc_score(input_y_argmax.reshape(-1), ndf_pred.reshape(-1))
    logger.info("END")
    return se, df
