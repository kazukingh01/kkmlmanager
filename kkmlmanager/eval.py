import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, log_loss
from kklogger import set_logger

# local package
from .util.numpy import NdarrayWithErr
from .calibration import expected_calibration_error, expected_calibration_error_per_ndata
LOGGER = set_logger(__name__)


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
    LOGGER.info("START")
    assert isinstance(input, np.ndarray)
    assert len(input.shape) == 2
    df = pd.DataFrame(index=np.arange(input.shape[0]))
    if func_predict is None:
        if   hasattr(model, "predict_proba"): func_predict = "predict_proba"
        elif hasattr(model, "predict"):       func_predict = "predict"
    LOGGER.info(f"model: {model}, is_reg: {is_reg}, func_predict: {func_predict}, kwargs: {kwargs}")
    assert isinstance(func_predict, str)    
    output = getattr(model, func_predict)(input, **kwargs)
    if isinstance(output, NdarrayWithErr):
        output = output.val
    assert isinstance(output, (np.ndarray)), f"type: {type(output)}"
    assert len(output.shape) in [1, 2]
    if len(output.shape) == 1: output = output.reshape(-1, 1)
    if is_reg:
        df[[f"predict_{i}" for i in range(output.shape[-1])]] = output
        LOGGER.info(f"we define column name 'predict' as 'predict_0'.")
        df["predict"] = df["predict_0"].copy()
    else:
        if output.shape[-1] == 1:
            output = np.concatenate([(1 - output), output], axis=1)
        if hasattr(model, "classes_") and isinstance(model.classes_, np.ndarray):
            ndf_class = model.classes_ 
            if ndf_class.shape[0] != output.shape[-1]:
                LOGGER.warning(f"shape is different. output: {output.shape}, model.classes_: {ndf_class.shape}")
                ndf_class = np.arange(output.shape[1])
        else:
            ndf_class = np.arange(output.shape[1])
        assert int(ndf_class.sum()) == int(np.arange(ndf_class.shape[0]).sum()) # allow [0, 1, 2, ...]. Don't allow [0, 2, 3, ...]
        df[[f"predict_proba_{i}" for i in ndf_class.astype(int)]] = output
        df["predict"] = np.argmax(output, axis=1)
    LOGGER.info("END")
    return df

def eval_model(input_x: np.ndarray, input_y: np.ndarray, model=None, is_reg: bool=False, func_predict: str=None, **kwargs):
    LOGGER.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    LOGGER.info(f"model: {model}, is_reg: {is_reg}, func_predict: {func_predict}, kwargs: {kwargs}")
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
        assert len(input_y.shape) in [1, 2]
        if len(input_y.shape) == 1:
            df["answer"] = input_y
            se["r2"]   = r2_score(input_y, df["predict"])
            se["rmse"] = np.sqrt( ((input_y - df["predict"].to_numpy()) ** 2).sum() / input_y.shape[0] )
            se["mae"]  = (np.abs(input_y - df["predict"].to_numpy())).sum() / input_y.shape[0]
        else:
            for i in range(input_y.shape[1]):
                df[f"answer_{i}"] = input_y[:, i]
                se[f"r2_{i}"]   = r2_score(input_y[:, i], df[f"predict_{i}"].to_numpy())
                se[f"rmse_{i}"] = np.sqrt( ((input_y[:, i] - df[f"predict_{i}"].to_numpy()) ** 2).sum() / input_y.shape[0] )
                se[f"mae_{i}"]  = (np.abs(input_y[:, i] - df[f"predict_{i}"].to_numpy())).sum() / input_y.shape[0]
    else:
        assert len(input_y.shape) in [1,2]
        n_class  = df.columns.str.contains("^predict_proba_", regex=True).sum()
        ndf_pred = df[[f"predict_proba_{i_class}" for i_class in np.arange(n_class, dtype=int)]].values
        if len(input_y.shape) == 1:
            df["answer"] = input_y
        else:
            if ndf_pred.shape[-1] < input_y.shape[-1]:
                LOGGER.warning(f"shape is different. output: {ndf_pred.shape}, answer: {input_y.shape}")
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
        se["auc_multi"]   = (se.loc[se.index.str.contains("^auc_")].values * weight).sum() / weight.sum()
        se["auc_all"]     = roc_auc_score(input_y_argmax.reshape(-1), ndf_pred.reshape(-1))
        se["ece_025"]     = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=0.25, is_consider_all_class=False)
        se["ece_050"]     = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=0.50, is_consider_all_class=False)
        se["ece_100"]     = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=1.00, is_consider_all_class=False)
        se["ece_200"]     = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=2.00, is_consider_all_class=False)
        se["ece_400"]     = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=4.00, is_consider_all_class=False)
        se["ece_025_all"] = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=0.25, is_consider_all_class=True)
        se["ece_050_all"] = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=0.50, is_consider_all_class=True)
        se["ece_100_all"] = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=1.00, is_consider_all_class=True)
        se["ece_200_all"] = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=2.00, is_consider_all_class=True)
        se["ece_400_all"] = expected_calibration_error(ndf_pred, input_y_class, n_bins=200, npow=4.00, is_consider_all_class=True)
        for i_class in np.arange(n_class):
            se[f"ece_100_c{i_class}"] = expected_calibration_error(ndf_pred[:, i_class], (input_y_class == i_class).astype(int), n_bins=200, npow=1.00, is_consider_all_class=False)
        se[f"ece_100_cmean"] = np.mean([se.get(f"ece_100_c{i_class}", float("nan")) for i_class in np.arange(n_class)])
        if ndf_pred.shape[0] >= 10:
            se["ece_n10_1"] = expected_calibration_error_per_ndata(ndf_pred, input_y_class, n_data_per_bin=10,    is_consider_all_class=True)
        if ndf_pred.shape[0] >= 100:
            se["ece_n10_2"] = expected_calibration_error_per_ndata(ndf_pred, input_y_class, n_data_per_bin=100,   is_consider_all_class=True)
        if ndf_pred.shape[0] >= 1000:
            se["ece_n10_3"] = expected_calibration_error_per_ndata(ndf_pred, input_y_class, n_data_per_bin=1000,  is_consider_all_class=True)
    LOGGER.info("END")
    return se, df
