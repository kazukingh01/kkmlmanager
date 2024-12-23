import pandas as pd
import numpy as np
import torch
from functools import partial
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# local package
from kkmlmanager.regproc import RegistryProc
from kkmlmanager.util.dataframe import parallel_apply, astype_faster
from kkmlmanager.util.com import check_type_list
from kklogger import set_logger
logger = set_logger(__name__)


__all__ = [
    "get_features_by_variance",
    "corr_coef_gpu_2array",
    "corr_coef_pearson_2array_numpy",
    "corr_coef_pearson_2array",
    "corr_coef_spearman_2array",
    "corr_coef_kendall_2array",
    "get_features_by_correlation",
    "get_features_by_randomtree_importance",
    "get_features_by_adversarial_validation",
    "create_features_by_basic_method"
]


def get_features_by_variance(
    df: pd.DataFrame, cutoff: float=0.99, ignore_nan: bool=True, batch_size: int=128, n_jobs: int=1
):
    """
    Usage::
        >>> df = pd.DataFrame(index=np.arange(1000))
        >>> df["aa"] = 2
        >>> df["bb"] = 3
        >>> df.iloc[500:, 1] = float("nan")
        >>> df
            aa   bb
        0    2  3.0
        1    2  3.0
        ..  ..  ...
        998  2  NaN
        999  2  NaN
        [1000 rows x 2 columns]
        >>> get_features_by_variance(df, cutoff=0.9, ignore_nan=False, batch_size=128, n_jobs=1)
        aa     True
        bb    False
        dtype: bool
        >>> get_features_by_variance(df, cutoff=0.5, ignore_nan=False, batch_size=128, n_jobs=1)
        aa     True
        bb     True
        dtype: bool
        >>> get_features_by_variance(df, cutoff=0.9, ignore_nan=True, batch_size=128, n_jobs=1)
        aa     True
        bb     True
        dtype: bool
    """
    logger.info("START")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(cutoff, float) and 0.0 < cutoff <= 1.0
    assert isinstance(ignore_nan, bool)
    assert isinstance(n_jobs, int) and n_jobs >= 1
    assert isinstance(batch_size, int) and batch_size > 1 # NG: batch_size=1
    df       = df.copy()
    # In object without conversion to float, nan sort in np.sort does not come at the end
    columns  = df.columns.copy()
    df_sort  = pd.DataFrame(index=np.arange(df.shape[0]))
    dtypes   = df.dtypes.copy()
    list_obj = []
    for _cols in dtypes.unique():
        list_obj += parallel_apply(df.loc[:, dtypes.index[dtypes == _cols]], get_features_by_variance_func1, axis=0, batch_size=batch_size, n_jobs=n_jobs)
    df_sort  = pd.concat([pd.DataFrame(y, columns=x) for x, y in list_obj], axis=1, ignore_index=False)
    df_sort  = df_sort.loc[:, columns]
    nsize    = int(df_sort.shape[0] * cutoff) - 1
    if nsize == 0: raise Exception(f"nsize is zero.")
    sebool   = pd.Series(False, index=df_sort.columns)
    if not ignore_nan:
        for i in range(0, df_sort.shape[0] - nsize):
            sewk1  = df_sort.iloc[i        , :]
            sewk2  = df_sort.iloc[i + nsize, :]
            boolwk = ((sewk1.isna() == sewk2.isna()) & sewk1.isna())
            boolwk = boolwk | (sewk1 == sewk2)
            sebool = sebool | boolwk
    else:
        func1 = partial(get_features_by_variance_func2, cutoff=cutoff)
        list_obj = parallel_apply(df_sort, func1, axis=0, batch_size=batch_size, n_jobs=n_jobs)
        for x, _ in list_obj: sebool.loc[x.index] = x
    logger.info("END")
    return sebool
def get_features_by_variance_func1(x: pd.DataFrame):
    return [x.columns, np.sort(x.values, axis=0)]
def get_features_by_variance_func2(df: pd.DataFrame, cutoff: float=None):
    dfwk  = df.isna()
    se_n  = df.shape[0] - dfwk.sum(axis=0)
    seret = pd.Series(False, index=df.columns)
    for x in se_n.index:
        length = se_n.loc[x]
        nsize  = int(length * cutoff)
        if nsize == 0:
            seret.loc[x] = True
            break
        nsize  = nsize - 1
        sewk   = df.loc[:, x].copy()
        sewk   = sewk[~dfwk.loc[:, x]]
        for i in range(0, length - nsize):
            if sewk.iloc[i] == sewk.iloc[i + nsize]:
                seret.loc[x] = True
                break
    return seret, None

def corr_coef_pearson_gpu(input: np.ndarray, _dtype=torch.float16, min_n: int=10) -> np.ndarray:
    """
    ref: https://sci-pursuit.com/math/statistics/correlation-coefficient.html
    """
    logger.info("START")
    """
    >>> input
    array( [[ 2.,  1.,  3., nan, nan],
            [ 0.,  2.,  2., nan, -3.],
            [ 8.,  2.,  9., nan, -7.],
            [ 1., nan,  5.,  5., -1.]])
    """
    tensor_max = torch.from_numpy(np.nanmax(input, axis=0)).to(_dtype).to("cuda:0")
    tensor_min = torch.from_numpy(np.nanmin(input, axis=0)).to(_dtype).to("cuda:0")
    tensor_max = (tensor_max - tensor_min)
    tensor_max[tensor_max == 0] = float("inf") # To avoid division by zero
    tens = torch.from_numpy(input).to(_dtype).to("cuda:0")
    tens = (tens - tensor_min) / tensor_max
    """
    >>> tensor_min
    tensor([ 0.,  1.,  2.,  5., -7.], device='cuda:0', dtype=torch.float16)
    >>> tensor_max
    tensor([8., 1., 7., inf, 6.], device='cuda:0', dtype=torch.float16)
    >>> tens
    tensor([[0.2500, 0.0000, 0.1428,    nan,    nan],
            [0.0000, 1.0000, 0.0000,    nan, 0.6665],
            [1.0000, 1.0000, 1.0000,    nan, 0.0000],
            [0.1250,    nan, 0.4285, 0.0000, 1.0000]], device='cuda:0', dtype=torch.float16)
    """
    tens_mean = torch.nanmean(tens, dim=0)
    tens      = tens - tens_mean
    """
    >>> tens
    tensor([[-0.0938, -0.6665, -0.2500,     nan,     nan],
            [-0.3438,  0.3335, -0.3928,     nan,  0.1113],
            [ 0.6562,  0.3335,  0.6074,     nan, -0.5552],
            [-0.2188,     nan,  0.0356,  0.0000,  0.4448]], device='cuda:0', dtype=torch.float16)
    """
    tens_nan = torch.isnan(tens)
    """
    >>> tens_nan
    tensor([[False, False, False,  True,  True],
            [False, False, False,  True, False],
            [False, False, False,  True, False],
            [False,  True, False, False, False]], device='cuda:0')
    """
    tens[tens_nan] = 0
    tens_n_Sx  = (~tens_nan).sum(dim=0)
    tens_nan   = (~tens_nan).to(torch.float16)
    tens_n_Sxy = torch.mm(tens_nan.t(), tens_nan)
    """
    >>> tens_n_Sxy
    tensor([[4., 3., 4., 1., 3.],
            [3., 3., 3., 0., 2.],
            [4., 3., 4., 1., 3.],
            [1., 0., 1., 1., 1.],
            [3., 2., 3., 1., 3.]], device='cuda:0', dtype=torch.float16)
    >>> tens_n_Sx
    tensor([4, 3, 4, 1, 3], device='cuda:0')
    """
    tens_Sxy  = torch.mm(tens.t(), tens)
    tens_Sx   = tens_Sxy[torch.eye(tens_Sxy.shape[0]).to(torch.bool)]
    tens_Sx   = (tens_Sx / tens_n_Sx).pow(1/2).unsqueeze(0)
    tens_SxSy = torch.mm(tens_Sx.t(), tens_Sx)
    tens_Sxy  = tens_Sxy / tens_n_Sxy
    """
    >>> tens_Sxy
    tensor([[ 0.1514,  0.0556,  0.1373,  0.0000, -0.1666],
            [ 0.0556,  0.2222,  0.0794,     nan, -0.0740],
            [ 0.1373,  0.0794,  0.1467,  0.0000, -0.1216],
            [ 0.0000,     nan,  0.0000,  0.0000,  0.0000],
            [-0.1666, -0.0740, -0.1216,  0.0000,  0.1729]], device='cuda:0', dtype=torch.float16)
    >>> tens_SxSy
    tensor([[0.1515, 0.1835, 0.1490, 0.0000, 0.1617],
            [0.1835, 0.2223, 0.1805, 0.0000, 0.1960],
            [0.1490, 0.1805, 0.1467, 0.0000, 0.1593],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.1617, 0.1960, 0.1593, 0.0000, 0.1729]], device='cuda:0', dtype=torch.float16)
    """
    tens_corr = tens_Sxy / tens_SxSy
    tens_corr = torch.nan_to_num(tens_corr, nan=0, posinf=0, neginf=0)
    """
    >>> tens_corr
    tensor([[ 0.9990,  0.3030,  0.9214,  0.0000, -1.0303],
            [ 0.3030,  0.9995,  0.4399,  0.0000, -0.3777],
            [ 0.9214,  0.4399,  1.0000,  0.0000, -0.7637],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [-1.0303, -0.3777, -0.7637,  0.0000,  1.0000]], device='cuda:0', dtype=torch.float16)
    """
    tens_corr[tens_n_Sxy <= min_n] = float("nan")
    """
    >>> tens_corr # min_n = 2
    tensor([[ 0.9990,  0.3030,  0.9214,     nan, -1.0303],
            [ 0.3030,  0.9995,  0.4399,     nan,     nan],
            [ 0.9214,  0.4399,  1.0000,     nan, -0.7637],
            [    nan,     nan,     nan,     nan,     nan],
            [-1.0303,     nan, -0.7637,     nan,  1.0000]], device='cuda:0', dtype=torch.float16)
    """
    logger.info("END")
    return tens_corr.cpu().numpy()

def corr_coef_pearson_2array_numpy(input_x: np.ndarray, input_y: np.ndarray, _dtype=np.float32, min_n: int=10) -> np.ndarray:
    """
    Faster than corr_coef_pearson_2array(..., is_gpu=False)
    """
    logger.info("START")
    assert input_x.shape[0] == input_y.shape[0]
    input_x, input_y = input_x.astype(_dtype), input_y.astype(_dtype)
    ndf = []
    for input in [input_x, input_y]:
        ndf_max = np.nanmax(input, axis=0)
        ndf_min = np.nanmin(input, axis=0)
        ndf_max = (ndf_max - ndf_min)
        ndf_max[ndf_max == 0] = float("inf") # To avoid division by zero
        ndf.append(input)
        ndf[-1] = (ndf[-1] - ndf_min) / ndf_max
    ndf_x, ndf_y = ndf
    ndf_x_mean   = np.nanmean(ndf_x, axis=0)
    ndf_y_mean   = np.nanmean(ndf_y, axis=0)
    ndf_x        = ndf_x - ndf_x_mean
    ndf_y        = ndf_y - ndf_y_mean
    ndf_x_nan    = np.isnan(ndf_x)
    ndf_y_nan    = np.isnan(ndf_y)
    ndf_x[ndf_x_nan] = 0
    ndf_y[ndf_y_nan] = 0
    ndf_n_Sx  = (~ndf_x_nan).sum(axis=0)
    ndf_n_Sy  = (~ndf_y_nan).sum(axis=0)
    ndf_x_nan = (~ndf_x_nan).astype(np.int32)
    ndf_y_nan = (~ndf_y_nan).astype(np.int32)
    ndf_n_Sxy = ndf_x_nan.T @ ndf_y_nan
    ndf_Sxy   = ndf_x.T     @ ndf_y
    ndf_Sxy   = ndf_Sxy / ndf_n_Sxy
    ndf_Sx    = np.sqrt(np.power(ndf_x, 2).sum(axis=0) / ndf_n_Sx).reshape(1, -1)
    ndf_Sy    = np.sqrt(np.power(ndf_y, 2).sum(axis=0) / ndf_n_Sy).reshape(1, -1)
    ndf_SxSy  = ndf_Sx.T @ ndf_Sy
    ndf_corr  = ndf_Sxy / ndf_SxSy
    ndf_corr  = np.nan_to_num(ndf_corr, nan=0, posinf=0, neginf=0)
    ndf_corr[ndf_n_Sxy <= min_n] = float("nan")
    logger.info("END")
    return ndf_corr

def corr_coef_pearson_2array(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float16", min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    logger.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert input_x.shape[0] == input_y.shape[0]
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: logger.warning("Note that the calculation is 'very SLOW'.")
    device = "cuda:0" if is_gpu else "cpu"
    tens = []
    for input in [input_x, input_y]:
        input = input.astype(getattr(np, dtype))
        tensor_max = torch.from_numpy(np.nanmax(input, axis=0)).to(getattr(torch, dtype)).to(device)
        tensor_min = torch.from_numpy(np.nanmin(input, axis=0)).to(getattr(torch, dtype)).to(device)
        tensor_max = (tensor_max - tensor_min)
        tensor_max[tensor_max == 0] = float("inf") # To avoid division by zero
        tens.append(torch.from_numpy(input).to(getattr(torch, dtype)).to(device))
        tens[-1] = (tens[-1] - tensor_min) / tensor_max
    tens_x, tens_y = tens
    tens_x_mean    = torch.nanmean(tens_x, dim=0)
    tens_y_mean    = torch.nanmean(tens_y, dim=0)
    tens_x         = tens_x - tens_x_mean
    tens_y         = tens_y - tens_y_mean
    tens_x_nan     = torch.isnan(tens_x)
    tens_y_nan     = torch.isnan(tens_y)
    tens_x[tens_x_nan] = 0
    tens_y[tens_y_nan] = 0
    tens_n_Sx  = (~tens_x_nan).sum(dim=0)
    tens_n_Sy  = (~tens_y_nan).sum(dim=0)
    tens_x_nan = (~tens_x_nan).to(torch.float16)
    tens_y_nan = (~tens_y_nan).to(torch.float16)
    tens_n_Sxy = torch.mm(tens_x_nan.t(), tens_y_nan)
    tens_Sxy   = torch.mm(tens_x.t(), tens_y)
    tens_Sxy   = tens_Sxy / tens_n_Sxy
    tens_Sx    = (tens_x.pow(2).sum(dim=0) / tens_n_Sx).pow(1/2).unsqueeze(0)
    tens_Sy    = (tens_y.pow(2).sum(dim=0) / tens_n_Sy).pow(1/2).unsqueeze(0)
    tens_SxSy  = torch.mm(tens_Sx.t(), tens_Sy)
    tens_corr  = tens_Sxy / tens_SxSy
    tens_corr  = torch.nan_to_num(tens_corr, nan=0, posinf=0, neginf=0)
    tens_corr[tens_n_Sxy < min_n] = float("nan")
    logger.info("END")
    return tens_corr.cpu().numpy()

def corr_coef_spearman_2array(df_x: pd.DataFrame, df_y: pd.DataFrame, is_to_rank: bool=True, n_jobs: int=1, dtype: str="float32", min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    logger.info("START")
    assert isinstance(df_x, pd.DataFrame)
    assert isinstance(df_y, pd.DataFrame)
    assert isinstance(is_to_rank, bool)
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: logger.warning("Note that the calculation is 'very SLOW'.")
    device = "cuda:0" if is_gpu else "cpu"
    if is_to_rank:
        assert isinstance(n_jobs, int) and n_jobs >= 1
        df_x = parallel_apply(df_x.copy(), corr_coef_spearman_2array_func1, axis=0, func_aft=lambda x,y,z: pd.concat(x, axis=1, ignore_index=False, sort=False).loc[y, z], batch_size=10, n_jobs=n_jobs)
        df_y = parallel_apply(df_y.copy(), corr_coef_spearman_2array_func1, axis=0, func_aft=lambda x,y,z: pd.concat(x, axis=1, ignore_index=False, sort=False).loc[y, z], batch_size=10, n_jobs=n_jobs)
    input_x, input_y = df_x.values, df_y.values
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert input_x.shape[1] >= input_y.shape[1]
    tens_x       = torch.from_numpy(input_x.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y_tmp   = torch.from_numpy(input_y.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y       = torch.zeros_like(tens_x, dtype=getattr(torch, dtype), device=device)
    tens_y[:, :tens_y_tmp.shape[1]] = tens_y_tmp
    tens_x_nonan = ~torch.isnan(tens_x)
    tens_y_nonan = ~torch.isnan(tens_y)
    tens_x       = tens_x / tens_x_nonan.sum(dim=0)
    tens_y       = tens_y / tens_y_nonan.sum(dim=0)
    tens_x[~tens_x_nonan] = 0
    tens_y[~tens_y_nonan] = 0
    # to rank
    tens_corr = torch.zeros(tens_x.shape[1], tens_y.shape[1]).to(torch.float32).to(device)
    tens_eye  = torch.eye(  tens_x.shape[1], tens_y.shape[1]).to(bool).to(device)
    for i in np.arange(tens_x.shape[1]):
        tens_diff = torch.roll(tens_x,       i, 1) - tens_y
        tens_bool = torch.roll(tens_x_nonan, i, 1) & tens_y_nonan
        tens_diff[~tens_bool] = 0
        tens_n = tens_bool.sum(dim=0)
        tenswk = 1 - ((tens_diff.pow(2).sum(dim=0) * 6) / (tens_n - (1/tens_n)))
        tens_corr[torch.roll(tens_eye, i, 1)] = torch.roll(tenswk, -i, 0)
    tens_nonan = torch.mm(tens_x_nonan.t().to(torch.float), tens_y_nonan.to(torch.float))
    tens_corr[tens_nonan < min_n] = float("nan")
    logger.info("END")
    return tens_corr[:, :tens_y_tmp.shape[1]].cpu().numpy()
def corr_coef_spearman_2array_func1(x):
    return x.rank(method="average")

def corr_coef_kendall_2array(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float16", n_sample: int=1000, n_iter: int=1, min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    logger.info("START")
    device = "cuda:0" if is_gpu else "cpu"
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert input_x.shape[1] >= input_y.shape[1]
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: logger.warning("Note that the calculation is 'very SLOW'.")
    device        = "cuda:0" if is_gpu else "cpu"
    tens_x        = torch.from_numpy(input_x.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y_tmp    = torch.from_numpy(input_y.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y        = torch.zeros_like(tens_x, dtype=getattr(torch, dtype), device=device)
    tens_y[:, :tens_y_tmp.shape[1]] = tens_y_tmp
    tens_x_nonan  = ~torch.isnan(tens_x)
    tens_y_nonan  = ~torch.isnan(tens_y)
    n_sample      = min(n_sample, tens_x.shape[0])
    tens_eye      = torch.eye(n_sample, n_sample, dtype=torch.bool, device=device)
    tens_eye      = torch.repeat_interleave(tens_eye.unsqueeze(0), tens_x.shape[1], 0)
    tens_eye      = torch.repeat_interleave(tens_eye.unsqueeze(0), tens_x.shape[1], 0) # 2 times
    tens_corr     = torch.zeros(tens_x.shape[1], tens_y.shape[1]).to(torch.float32).to(device)
    tens_corr_n   = torch.zeros(tens_x.shape[1], tens_y.shape[1]).to(torch.float32).to(device)
    ndf_index     = np.arange(tens_x.shape[0])
    for _ in range(n_iter):
        logger.info(f"iter: {_}")
        index         = np.random.permutation(ndf_index)
        _tens_x       = tens_x[index[:n_sample]]
        _tens_y       = tens_y[index[:n_sample]]
        _tens_x_nonan = tens_x_nonan[index[:n_sample]]
        _tens_y_nonan = tens_y_nonan[index[:n_sample]]
        h, w = _tens_x.shape
        _tens_x_all       = _tens_x.T.      reshape(w, 1, h) > _tens_x.T.      reshape(w, h, 1)
        _tens_x_nonan_all = _tens_x_nonan.T.reshape(w, 1, h) & _tens_x_nonan.T.reshape(w, h, 1)
        h, w = _tens_y.shape
        _tens_y_all       = _tens_y.T.      reshape(w, 1, h) > _tens_y.T.      reshape(w, h, 1)
        _tens_y_nonan_all = _tens_y_nonan.T.reshape(w, 1, h) & _tens_y_nonan.T.reshape(w, h, 1)
        index_x = torch.arange(_tens_x.shape[-1])
        index_x = torch.stack([torch.roll(index_x, i, 0) for i in range(index_x.shape[0])])
        index_y = torch.arange(_tens_y.shape[-1]).repeat(_tens_y.shape[-1]).reshape(-1, _tens_y.shape[-1])
        tens_bool       = _tens_x_all[index_x]      == _tens_y_all[index_y]
        tens_nonan_bool = _tens_x_nonan_all[index_x] & _tens_y_nonan_all[index_y]
        tens_bool       = tens_bool.to(torch.int8)
        tens_bool[tens_eye] = -1
        tens_bool[~tens_nonan_bool] = -1
        tens_nonan_bool[tens_eye] = False
        tens_diff = (tens_bool == 1).sum(dim=(-1,-2)) - (tens_bool == 0).sum(dim=(-1,-2))
        tens_n    = tens_nonan_bool.sum(dim=(-1,-2)).to(torch.int32)
        tens_diff = tens_diff / tens_n
        tens_diff = torch.nan_to_num(tens_diff, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        tens_diff[tens_n < min_n] = float("nan")
        _isnan       = ~torch.isnan(tens_diff)
        tens_corr_n += _isnan
        tens_diff[~_isnan] = 0
        tens_corr   += tens_diff
    tens_corr = tens_corr / tens_corr_n
    tens_corr = tens_corr[:, :tens_y_tmp.shape[1]].cpu().numpy()
    tens_corr = np.nan_to_num(tens_corr, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
    logger.info("END")
    return tens_corr

def corr_coef_kendall_2array_numpy(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float32", n_sample: int=1000, n_iter: int=1, min_n: int=10) -> np.ndarray:
    logger.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert input_x.shape == input_y.shape
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    ndf_x = input_x.astype(getattr(np, dtype))
    ndf_y = input_y.astype(getattr(np, dtype))
    ndf_x_nonan  = ~np.isnan(ndf_x)
    ndf_y_nonan  = ~np.isnan(ndf_y)
    n_sample     = min(n_sample, ndf_x.shape[0])
    ndf_eye      = np.eye(n_sample, n_sample, dtype=bool)
    ndf_eye      = np.tile(ndf_eye.reshape(1, *ndf_eye.shape), (ndf_x.shape[1], 1, 1))
    ndf_eye_corr = np.eye(ndf_x.shape[1], ndf_x.shape[1], dtype=bool)
    ndf_corr     = np.zeros((ndf_x.shape[1], ndf_y.shape[1]))
    ndf_corr_n   = np.zeros((ndf_x.shape[1], ndf_y.shape[1]))
    ndf_index    = np.arange(ndf_x.shape[0])
    for _ in range(n_iter):
        logger.info(f"iter: {_}")
        index         = np.random.permutation(ndf_index)
        _ndf_x       = ndf_x[index[:n_sample]]
        _ndf_y       = ndf_y[index[:n_sample]]
        _ndf_x_nonan = ndf_x_nonan[index[:n_sample]]
        _ndf_y_nonan = ndf_y_nonan[index[:n_sample]]
        h, w = _ndf_x.shape
        _ndf_x_all       = _ndf_x.T.      reshape(w, 1, h) > _ndf_x.T.      reshape(w, h, 1)
        _ndf_x_nonan_all = _ndf_x_nonan.T.reshape(w, 1, h) & _ndf_x_nonan.T.reshape(w, h, 1)
        h, w = _ndf_y.shape
        _ndf_y_all       = _ndf_y.T.      reshape(w, 1, h) > _ndf_y.T.      reshape(w, h, 1)
        _ndf_y_nonan_all = _ndf_y_nonan.T.reshape(w, 1, h) & _ndf_y_nonan.T.reshape(w, h, 1)
        for i in np.arange(_ndf_x_all.shape[0]):
            ndf_bool       = (np.roll(_ndf_x_all,       i, axis=0) == _ndf_y_all)
            ndf_nonan_bool = (np.roll(_ndf_x_nonan_all, i, axis=0) &  _ndf_y_nonan_all)
            ndf_bool       = ndf_bool.astype(np.int8)
            ndf_bool[ndf_eye] = -1
            ndf_bool[~ndf_nonan_bool] = -1
            ndf_nonan_bool[ndf_eye] = False
            ndf_diff = (ndf_bool == 1).sum(axis=-1).sum(axis=-1) - (ndf_bool == 0).sum(axis=-1).sum(axis=-1)
            ndf_n    = ndf_nonan_bool.sum(axis=-1).sum(axis=-1)
            ndf_diff = ndf_diff / ndf_n
            ndf_diff = np.nan_to_num(ndf_diff, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
            ndf_diff[ndf_n < min_n] = float("nan")
            _isnan    = ~np.isnan(ndf_diff)
            _ndf_eye = np.roll(ndf_eye_corr, i, 1)
            ndf_corr_n[_ndf_eye] += np.roll(_isnan,    -i, 0)
            ndf_diff[~_isnan] = 0
            ndf_corr[  _ndf_eye] += np.roll(ndf_diff, -i, 0)
    ndf_corr = ndf_corr / ndf_corr_n
    logger.info("END")
    return ndf_corr

def get_features_by_correlation(
    df: pd.DataFrame, dtype: str="float16", is_gpu: bool=False, 
    corr_type: str="pearson", batch_size: int=100, min_n: int=10, n_jobs: int=1,
    **kwargs
) -> pd.DataFrame:
    logger.info("START")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(is_gpu, bool)
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(corr_type, str) and corr_type in ["pearson", "spearman", "kendall"]
    assert isinstance(min_n,      int) and min_n > 0
    assert isinstance(batch_size, int) and batch_size >= 1
    assert isinstance(n_jobs,     int) and n_jobs >= 1
    df_corr = pd.DataFrame(float("nan"), index=df.columns, columns=df.columns)
    batch_size = min(df.shape[1], batch_size)
    if batch_size == 1:
        batch = [np.arange(df.shape[1])]
    else:
        batch = np.array_split(np.arange(df.shape[1]), df.shape[1] // batch_size)
    logger.info("convert to astype...")
    df = astype_faster(df.copy(), list_astype=[{"from": None, "to": getattr(np, dtype)}], batch_size=10, n_jobs=n_jobs)
    if corr_type == "spearman":
        df = parallel_apply(
            df.copy(), corr_coef_spearman_2array_func1, axis=0, 
            func_aft=lambda x,y,z: pd.concat(x, axis=1, ignore_index=False, sort=False).loc[y, z], 
            batch_size=10, n_jobs=n_jobs
        )
    if is_gpu:
        logger.info(f"calculate correlation [GPU] corr_type: {corr_type}...")
        n_iter = int(((len(batch) ** 2 - len(batch)) / 2) + len(batch))
        i_iter = 0
        for i, batch_x in enumerate(batch):
            for batch_y in batch[i:]:
                i_iter += 1
                logger.info(f"iter: {i_iter} / {n_iter}")
                df_x, df_y = df.iloc[:, batch_x], df.iloc[:, batch_y]
                input_x, input_y = df_x.values, df_y.values
                if corr_type == "pearson":
                    ndf_corr = corr_coef_pearson_2array(input_x, input_y, dtype=dtype, min_n=min_n, is_gpu=is_gpu)
                elif corr_type == "spearman":
                    ndf_corr = corr_coef_spearman_2array(df_x, df_y, is_to_rank=False, dtype=dtype, min_n=min_n, is_gpu=is_gpu)
                elif corr_type == "kendall":
                    dictwk = {x:y for x, y in kwargs.items() if x in ["n_sample", "n_iter"]}
                    ndf_corr = corr_coef_kendall_2array(input_x, input_y, dtype=dtype, min_n=min_n, is_gpu=is_gpu, **dictwk)
                df_corr.iloc[batch_x, batch_y] = ndf_corr
    else:
        logger.info(f"calculate correlation [CPU] corr_type: {corr_type}...")
        if corr_type == "pearson":
            func1    = partial(get_features_by_correlation_func1, dtype=dtype, min_n=min_n)
            list_obj = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([
                delayed(func1)(df.iloc[:, batch_x], df.iloc[:, batch_y], (batch_x, batch_y))
                for i, batch_x in enumerate(batch) for batch_y in batch[i:]
            ])
        else:
            logger.raise_error(f"corr_type: {corr_type} is not supported with CPU")
        for (batch_x, batch_y), ndf_corr in list_obj:
            df_corr.iloc[batch_x, batch_y] = ndf_corr
    logger.info("END")
    return df_corr
def get_features_by_correlation_func1(input_x, input_y, batch_xy: tuple, dtype=None, min_n=None):
    input_x, input_y = input_x.values.astype(np.float32), input_y.values.astype(np.float32)
    ndf_corr = corr_coef_pearson_2array_numpy(input_x, input_y, _dtype=getattr(np, dtype), min_n=min_n)
    return batch_xy, ndf_corr

def get_sklearn_trees_model_info(model) -> pd.DataFrame:
    logger.info("START")
    df = pd.DataFrame(np.concatenate([x.tree_.feature  for x in model.estimators_]), columns=["i_feature"])
    df["n_sample"]       = np.concatenate([x.tree_.n_node_samples                         for x in model.estimators_])
    df["n_sample_left"]  = np.concatenate([x.tree_.n_node_samples[x.tree_.children_left]  for x in model.estimators_])
    df["n_sample_right"] = np.concatenate([x.tree_.n_node_samples[x.tree_.children_right] for x in model.estimators_])
    df["impurity"]       = np.concatenate([x.tree_.impurity                         for x in model.estimators_])
    df["impurity_left"]  = np.concatenate([x.tree_.impurity[x.tree_.children_left]  for x in model.estimators_])
    df["impurity_right"] = np.concatenate([x.tree_.impurity[x.tree_.children_right] for x in model.estimators_])
    df["importance"]     = df["impurity"] * df["n_sample"] - ((df["impurity_left"] * df["n_sample_left"]) + (df["impurity_right"] * df["n_sample_right"]))
    df = df.loc[df["i_feature"] >= 0]
    logger.info("END")
    return df

def get_features_by_randomtree_importance(
    df: pd.DataFrame, columns_exp: list[str], columns_ans: str, dtype=np.float32, batch_size: int=25,
    is_reg: bool=False, max_iter: int=1, min_count: int=100, n_jobs: int=1, 
    **kwargs
) -> pd.DataFrame:
    logger.info("START")
    assert isinstance(df, pd.DataFrame)
    assert check_type_list(columns_exp, str)
    assert isinstance(columns_ans, str)
    assert isinstance(is_reg, bool)
    assert dtype in [float, np.float16, np.float32, np.float64]
    assert isinstance(batch_size, int) and batch_size > 0
    # row
    regproc_df = RegistryProc(n_jobs=n_jobs)
    regproc_df.register("ProcDropNa", columns_ans)
    if not is_reg: regproc_df.register("ProcCondition", f"{columns_ans} >= 0")
    df = regproc_df.fit(df[columns_exp + [columns_ans]])
    # columns explain
    regproc_exp = RegistryProc(n_jobs=n_jobs)
    regproc_exp.register("ProcAsType", dtype, batch_size=batch_size)
    regproc_exp.register("ProcToValues")
    regproc_exp.register("ProcReplaceInf", posinf=float("nan"), neginf=float("nan"))
    regproc_exp.register("ProcFillNaMinMax")
    regproc_exp.register("ProcFillNa", 0)
    ndf_x = regproc_exp.fit(df[columns_exp])
    # columns answer
    regproc_ans = RegistryProc(n_jobs=n_jobs)
    if is_reg: regproc_ans.register("ProcAsType", np.float32)
    else:      regproc_ans.register("ProcAsType", np.int32)
    regproc_ans.register("ProcToValues")
    regproc_ans.register("ProcReshape", (-1, ))
    ndf_y = regproc_ans.fit(df[[columns_ans]])
    logger.info(f"\ncolumns_exp: {columns_exp[:10]}\ninput:{ndf_x.shape}\ncolumns_ans: {columns_ans}, target:{ndf_y.shape}")
    se_cnt = pd.Series(0, index=np.arange(len(columns_exp)), dtype=float, name="count")
    se_imp = pd.Series(0, index=np.arange(len(columns_exp)), dtype=float, name="importance")
    i = 0
    for i in range(max_iter):
        logger.info(f"create forest. loop: {i}, cnt: {se_cnt.median()}, max: {se_cnt.max()}")
        dictwk = {
            "bootstrap":False, "n_estimators": max(n_jobs*10, 100), "max_depth": None, "max_features":"sqrt", "class_weight": "balanced",
            "min_samples_split":int(np.log2(ndf_x.shape[0])), "verbose":3, "random_state":i, "n_jobs": n_jobs
        }
        for x, y in kwargs.items():
            if x in ["bootstrap", "n_estimators", "max_depth", "max_features", "verbose", "min_samples_split", "criterion", "class_weight"]: dictwk[x] = y
        if is_reg:
            for x in ["class_weight"]:
                if x in dictwk: del dictwk[x]
            model = ExtraTreesRegressor( **dictwk)
        else:
            model = ExtraTreesClassifier(**dictwk)
        if i == 0: logger.info(f"model: {model}")
        model.fit(ndf_x, ndf_y)
        dfwk    = get_sklearn_trees_model_info(model)
        se_cnt += dfwk.groupby("i_feature").size()
        se_imp += dfwk.groupby("i_feature")["importance"].sum()
        if se_cnt.median() >= min_count: break
    df_feadtures = pd.concat([se_cnt, se_imp], axis=1, ignore_index=False, sort=False)
    df_feadtures.index = columns_exp
    df_feadtures["ratio"] = (df_feadtures["importance"] / df_feadtures["count"])
    logger.info("END")
    return df_feadtures

def get_features_by_adversarial_validation(
    df_train: pd.DataFrame, df_test: pd.DataFrame, columns_exp: list[str], columns_ans: str=None,
    n_split: int=5, n_cv: int=5, dtype=np.float32, batch_size: int=25, n_jobs: int=1, **kwargs
):
    logger.info("START")
    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_test,  pd.DataFrame)
    assert check_type_list(columns_exp, str)
    assert columns_ans is None or isinstance(columns_ans, str)
    if columns_ans is not None: columns_exp = columns_exp + [columns_ans]
    assert isinstance(n_split, int) and n_split >= 2
    assert isinstance(n_cv,    int) and n_cv    >= 1
    assert dtype in [float, np.float16, np.float32, np.float64]
    assert isinstance(batch_size, int) and batch_size > 0
    # row
    index_df = np.concatenate([df_train.index.values, df_test.index.values])
    # columns explain
    regproc_exp = RegistryProc(n_jobs=n_jobs)
    regproc_exp.register("ProcAsType", dtype, batch_size=batch_size)
    regproc_exp.register("ProcToValues")
    regproc_exp.register("ProcReplaceInf", posinf=float("nan"), neginf=float("nan"))
    regproc_exp.register("ProcFillNaMinMax")
    regproc_exp.register("ProcFillNa", 0)
    ndf_x1 = regproc_exp.fit(df_train[columns_exp].copy())
    ndf_x2 = regproc_exp(df_test[columns_exp].copy())
    ndf_x  = np.concatenate([ndf_x1, ndf_x2], axis=0)
    # columns answer
    ndf_y = np.concatenate([np.zeros(df_train.shape[0]), np.ones(df_test.shape[0])]).astype(int)
    logger.info(f"input: {ndf_x.shape}, target: {ndf_y.shape}")
    # model
    dictwk = {
        "bootstrap":False, "n_estimators": max(n_jobs*10, 100), "max_depth": None, "max_features":"sqrt",
        "min_samples_split":int(np.log2(ndf_x.shape[0])), "verbose":3, "random_state": 0, "n_jobs": n_jobs
    }
    for x, y in kwargs.items():
        if x in [
            "bootstrap", "n_estimators", "max_depth", "max_features", "verbose", 
            "min_samples_split", "criterion", "random_state"
        ]: dictwk[x] = y
    model = RandomForestClassifier(**dictwk)
    logger.info(f"model: {model}")
    # cross validation
    df_imp, df_pred = pd.DataFrame(), pd.DataFrame()
    for i_cv, (index_train, index_test) in enumerate(StratifiedKFold(n_splits=n_split).split(np.arange(ndf_x.shape[0], dtype=int), ndf_y)):
        logger.info(f"cross validation: {i_cv}")
        model.fit(ndf_x[index_train].copy(), ndf_y[index_train].copy())
        df_imp    = pd.concat([df_imp, get_sklearn_trees_model_info(model)], axis=0, ignore_index=True, sort=False)
        ndf_pred  = model.predict_proba(ndf_x[index_test])
        dfwk_pred = pd.DataFrame(np.argmax(ndf_pred, axis=1), columns=["predict"])
        dfwk_pred[[f"predict_proba_{i}" for i in range(ndf_pred.shape[-1])]] = ndf_pred
        dfwk_pred["answer"] = ndf_y[index_test]
        dfwk_pred["i_cv"]   = i_cv
        dfwk_pred["index"]  = index_df[index_test]
        df_pred = pd.concat([df_pred, dfwk_pred], axis=0, ignore_index=True, sort=False)
        if (n_cv - 1) <= i_cv: break
    df_adv = pd.DataFrame(index=np.arange(len(columns_exp)))
    df_adv["count"]      = df_imp.groupby("i_feature").size()
    df_adv["importance"] = df_imp.groupby("i_feature")["importance"].sum()
    df_adv["ratio"]      = (df_adv["importance"] / df_adv["count"])
    df_adv.index = columns_exp
    se_eval        = pd.Series(dtype=object)
    se_eval["auc"] = roc_auc_score( df_pred["answer"].values, df_pred["predict_proba_1"].values)
    se_eval["acc"] = accuracy_score(df_pred["answer"].values, df_pred["predict"].values)
    logger.info(f"roc_auc: {se_eval['auc']}, accuracy: {se_eval['acc']}")
    logger.info("END")
    return df_adv, df_pred, se_eval

def create_features_by_basic_method(
    df: pd.DataFrame, colname_begin: str, replace_inf: float=float("nan"),
    calc_list=["sum","mean","std","max","min","rank","diff","ratio"],
) -> pd.DataFrame:
    """
    Params::
        df: input
        colname_begin: Column name to be added at the beginning
        columns: 
        calc_list: ["sum","mean","std","max","min","rank","diff","ratio"]
    """
    logger.info(f"START column: {colname_begin}")
    assert isinstance(df, pd.DataFrame) and df.shape[1] >= 2
    assert isinstance(colname_begin, str)
    df_f = pd.DataFrame(index=df.index)
    if "sum"  in calc_list: df_f[colname_begin + "_sum"]  = df.sum(axis=1)
    if "mean" in calc_list: df_f[colname_begin + "_mean"] = df.mean(axis=1)
    if "std"  in calc_list: df_f[colname_begin + "_std"]  = df.std(axis=1)
    if "max"  in calc_list: df_f[colname_begin + "_max"]  = df.max(axis=1)
    if "min"  in calc_list: df_f[colname_begin + "_min"]  = df.min(axis=1)
    if "rank" in calc_list:
        dfwk = df.rank(axis=1, method="average")
        for x in dfwk.columns:
            df_f[colname_begin+"_"+x+"_rank"] = dfwk[x].astype(np.float16)
    for i, x in enumerate(df.columns):
        for y in df.columns[i+1:]:
            if "diff"  in calc_list: df_f[colname_begin+"_"+x+"_"+y+"_diff" ] =  df[x] - df[y]
            if "ratio" in calc_list: df_f[colname_begin+"_"+x+"_"+y+"_ratio"] = (df[x] / df[y]).astype(np.float32)
    if replace_inf is not None:
        df_f = df_f.replace(float("inf"), replace_inf).replace(float("-inf"), replace_inf)
    logger.info("END")
    return df_f
