import pandas as pd
import numpy as np
import polars as pl
try: import torch
except ImportError: torch = None
from functools import partial
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# local package
from .regproc import RegistryProc
from .util.dataframe import parallel_apply
from .util.com import check_type_list
from kklogger import set_logger
LOGGER = set_logger(__name__)


__all__ = [
    "get_features_by_variance",
    "get_features_by_variance_pl",
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
    LOGGER.info("START")
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
    LOGGER.info("END")
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

def get_features_by_variance_pl(df: pl.DataFrame, cutoff: float=0.99, ignore_nan: bool=True, n_divide: int=10000) -> pd.Series:
    assert isinstance(df, pl.DataFrame)
    assert isinstance(cutoff, float) and 0.0 < cutoff < 1.0
    assert isinstance(ignore_nan, bool)
    assert isinstance(n_divide, int) and n_divide >= 1000
    df = df.with_columns([
        pl.col(pl.Float32).fill_nan(None),
        pl.col(pl.Float64).fill_nan(None),
    ]).with_columns(pl.all().sort())
    n_data = df.shape[0]
    if ignore_nan:
        se_null = df.null_count()
        if np.any((se_null == n_data).to_numpy()):
            LOGGER.raise_error("Some columns have all NaN values. You should set ignore_nan=False first.", ValueError())
        dictwk  = {x.name: np.round(np.linspace(x[0], n_data - 1, n_divide)).astype(int) for x in se_null}
        df      = pl.concat([df[x][y].to_frame() for x, y in dictwk.items()], how="horizontal")
    else:
        idx = np.round(np.linspace(0, n_data - 1, n_divide)).astype(int)
        df  = df[idx]
    idx    = np.arange(n_divide, dtype=int)
    idx    = np.stack([idx, idx + int(n_divide * cutoff)]).T
    idx    = idx[np.sum(idx >= n_divide, axis=1) == 0]
    dfwk1  = df[idx[:, 0]]
    dfwk2  = df[idx[:, 1]]
    sebool = (dfwk1.with_columns([(pl.col(x) == dfwk2[x]) | (pl.col(x).is_null() & dfwk2[x].is_null()) for x in dfwk1.columns]).sum() > 0)
    return sebool.to_pandas().iloc[0]

def corr_coef_pearson_gpu(input: np.ndarray, _dtype: str | type="float16", min_n: int=10) -> np.ndarray:
    """
    ref: https://sci-pursuit.com/math/statistics/correlation-coefficient.html
    """
    LOGGER.info("START")
    """
    >>> input
    array( [[ 2.,  1.,  3., nan, nan],
            [ 0.,  2.,  2., nan, -3.],
            [ 8.,  2.,  9., nan, -7.],
            [ 1., nan,  5.,  5., -1.]])
    """
    if isinstance(_dtype, str):
        _dtype = getattr(torch, _dtype)
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
    LOGGER.info("END")
    return tens_corr.cpu().numpy()

def corr_coef_pearson_2array_numpy(input_x: np.ndarray, input_y: np.ndarray, dtype=np.float32, min_n: int=1000) -> np.ndarray:
    """
    Faster than corr_coef_pearson_2array(..., is_gpu=False)
    """
    LOGGER.info("START")
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert input_x.shape[0] < np.iinfo(np.int32).max
    input_x, input_y = input_x.astype(dtype), input_y.astype(dtype)
    ndf = []
    for input in [input_x, input_y]:
        input   = input - np.nanmin(input, axis=0).reshape(1, -1)
        ndf_max = np.nanmax(input, axis=0).reshape(1, -1)
        ndf_max[ndf_max < 1e-10] = float("inf") # To avoid division by zero
        input   = input / ndf_max
        ndf.append(input)
    ndf_x, ndf_y = ndf
    ndf_x_mean   = np.nanmean(ndf_x, axis=0)
    ndf_y_mean   = np.nanmean(ndf_y, axis=0)
    ndf_x        = ndf_x - ndf_x_mean.reshape(1, -1)
    ndf_y        = ndf_y - ndf_y_mean.reshape(1, -1)
    ndf_x_nan    = np.isnan(ndf_x)
    ndf_y_nan    = np.isnan(ndf_y)
    ndf_x[ndf_x_nan] = dtype(0)
    ndf_y[ndf_y_nan] = dtype(0)
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
    LOGGER.info("END")
    return ndf_corr

def corr_coef_pearson_2array(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float16", min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    LOGGER.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert input_x.shape[0] == input_y.shape[0]
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: LOGGER.warning("Note that the calculation is 'very SLOW'.")
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
    LOGGER.info("END")
    return tens_corr.cpu().numpy()

def corr_coef_spearman_2array_numpy(input_x: np.ndarray, input_y: np.ndarray, is_to_rank: bool=True, dtype=np.float32, min_n: int=10) -> np.ndarray:
    """
    When calculating the Spearman correlation coefficient with NaN values, rankings should be done on the values excluding the NaNs.
    However, calculating this between columns is too slow. Therefore, I first normalize the values to a 0-1 range and 
    then scale them back to the original range (excluding NaNs) to prevent divergence during normalization.
    """
    LOGGER.info("START")
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert isinstance(is_to_rank, bool)
    assert dtype in [np.float32, np.float64]
    assert isinstance(min_n, int) and min_n >= 0
    ndf_not_nan_x = (~np.isnan(input_x))
    ndf_not_nan_y = (~np.isnan(input_y))
    input_x = input_x.astype(dtype)
    input_y = input_y.astype(dtype)
    if is_to_rank:
        input_x = pl.from_numpy(input_x).fill_nan(None).with_columns(pl.all().rank(method="random")).to_numpy()
        input_y = pl.from_numpy(input_y).fill_nan(None).with_columns(pl.all().rank(method="random")).to_numpy()
    ndf = []
    for input in [input_x, input_y]:
        input   = input - np.nanmin(input, axis=0).reshape(1, -1)
        ndf_max = np.nanmax(input, axis=0).reshape(1, -1)
        ndf_max[ndf_max < 1e-10] = float("inf") # To avoid division by zero
        input   = input / ndf_max
        ndf.append(input)
    ndf_x, ndf_y = ndf
    list_corr = []
    for _ndf_y, _ndf_not_nan_y in zip(ndf_y.T, ndf_not_nan_y.T):
        ndf_n  = (ndf_not_nan_x & _ndf_not_nan_y.reshape(-1, 1)).sum(axis=0)
        _ndf_y = np.tile(_ndf_y, (ndf_x.shape[1], 1)).T
        _ndf_x =  ndf_x * ndf_n.reshape(1, -1)
        _ndf_y = _ndf_y * ndf_n.reshape(1, -1)
        list_corr.append(1 - (6 * np.nansum((_ndf_x - _ndf_y) ** 2, axis=0) / (ndf_n ** 3 - ndf_n)))
    ndf_not_nan = ndf_not_nan_x.T.astype(np.int32) @ ndf_not_nan_y.astype(np.int32)
    ndf_corr    = np.stack(list_corr).T.astype(dtype)
    assert ndf_corr.shape == ndf_not_nan.shape
    ndf_corr[ndf_not_nan <= min_n] = float("nan")
    LOGGER.info("END")
    return ndf_corr

def corr_coef_spearman_2array(df_x: pl.DataFrame, df_y: pl.DataFrame, is_to_rank: bool=True, dtype: str="float32", min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    LOGGER.info("START")
    assert isinstance(df_x, pl.DataFrame)
    assert isinstance(df_y, pl.DataFrame)
    assert isinstance(is_to_rank, bool)
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: LOGGER.warning("Note that the calculation is 'very SLOW'.")
    device = "cuda:0" if is_gpu else "cpu"
    if is_to_rank:
        df_x = df_x.fill_nan(None).with_columns(pl.all().rank(method="random"))
        df_y = df_y.fill_nan(None).with_columns(pl.all().rank(method="random"))
    input_x, input_y = df_x.to_numpy(), df_y.to_numpy()
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
    LOGGER.info("END")
    return tens_corr[:, :tens_y_tmp.shape[1]].cpu().numpy()

def corr_coef_kendall_2array(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float16", n_sample: int=1000, n_iter: int=1, min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    LOGGER.info("START")
    device = "cuda:0" if is_gpu else "cpu"
    assert isinstance(input_x, np.ndarray)
    assert isinstance(input_y, np.ndarray)
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert input_x.shape[1] >= input_y.shape[1]
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: LOGGER.warning("Note that the calculation is 'very SLOW'.")
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
        LOGGER.info(f"iter: {_}")
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
    LOGGER.info("END")
    return tens_corr

def corr_coef_kendall_2array_numpy(input_x: np.ndarray, input_y: np.ndarray, dtype: str="float32", n_sample: int=1000, n_iter: int=1, min_n: int=10) -> np.ndarray:
    LOGGER.info("START")
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
        LOGGER.info(f"iter: {_}")
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
    LOGGER.info("END")
    return ndf_corr

def corr_coef_chatterjee_2array(df_x: pl.DataFrame, df_y: pl.DataFrame, is_to_rank: bool=True, dtype: str="float32", min_n: int=10, is_gpu: bool=False) -> np.ndarray:
    LOGGER.info("START")
    assert isinstance(df_x, pl.DataFrame)
    assert isinstance(df_y, pl.DataFrame)
    assert isinstance(is_to_rank, bool)
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(min_n, int) and min_n >= 0
    assert isinstance(is_gpu, bool)
    if not is_gpu: LOGGER.warning("Note that the calculation is 'very SLOW'.")
    device = "cuda:0" if is_gpu else "cpu"
    if is_to_rank:
        df_x = df_x.fill_nan(None).with_columns(pl.all().rank(method="random"))
        df_y = df_y.fill_nan(None).with_columns(pl.all().rank(method="random"))
    input_x, input_y = df_x.to_numpy(), df_y.to_numpy()
    assert len(input_x.shape) == len(input_y.shape) == 2
    assert input_x.shape[0] == input_y.shape[0]
    assert input_x.shape[1] >= input_y.shape[1]
    tens_x       = torch.from_numpy(input_x.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y_tmp   = torch.from_numpy(input_y.astype(getattr(np, dtype))).to(getattr(torch, dtype)).to(device)
    tens_y       = torch.zeros_like(tens_x, dtype=getattr(torch, dtype), device=device)
    tens_y[:, :tens_y_tmp.shape[1]] = tens_y_tmp
    tens_x_nonan = ~torch.isnan(tens_x)
    tens_y_nonan = ~torch.isnan(tens_y)
    # to rank
    tens_corr = torch.zeros(tens_x.shape[1], tens_y.shape[1], dtype=torch.float32, device=tens_x.device)
    tens_eye  = torch.eye(  tens_x.shape[1], tens_y.shape[1], dtype=bool,          device=tens_x.device)
    col       = torch.arange(tens_y.size(1), device=tens_y.device).view(1, -1).expand_as(tens_y)
    for i in np.arange(tens_x.shape[1]):
        tens_roll = torch.roll(tens_x, i, 1)
        tens_bool = torch.roll(tens_x_nonan, i, 1) & tens_y_nonan
        tens_trgt = tens_y.clone()
        tens_trgt[~tens_bool] = float("nan")
        tens_trgt = 3.0 * tens_trgt / (torch.pow(tens_bool.sum(dim=0), 2) - 1) # scale first
        tens_idx  = torch.sort(tens_roll, dim=0).indices
        tens_trgt = tens_trgt.gather(dim=0, index=tens_idx)
        mask      = ~torch.isnan(tens_trgt)
        pos       = mask.cumsum(0) - 1
        tens_sort = torch.full_like(tens_trgt, float("nan"), dtype=tens_trgt.dtype, device=tens_trgt.device)
        tens_sort[pos[mask], col[mask]] = tens_trgt[mask]
        tens_diff = tens_sort[:-1, :] - tens_sort[1:, :]
        tens_diff[torch.isnan(tens_diff)] = 0.0
        tenswk    = 1 - torch.abs(tens_diff).sum(dim=0)
        tens_corr[torch.roll(tens_eye, i, 1)] = torch.roll(tenswk, -i, 0)
    tens_nonan = torch.mm(tens_x_nonan.t().to(torch.float), tens_y_nonan.to(torch.float))
    tens_corr[tens_nonan < min_n] = float("nan")
    LOGGER.info("END")
    return tens_corr[:, :tens_y_tmp.shape[1]].cpu().numpy()

def get_features_by_correlation(df: pl.DataFrame, dtype: str="float16", is_gpu: bool=False, corr_type: str="pearson", batch_size: int=100, min_n: int=10, n_jobs: int=1, **kwargs) -> pd.DataFrame:
    """
    Normal numpy pr polars corr method cannot consider Nan. So custom function is used in this code.
    """
    LOGGER.info("START")
    assert isinstance(df, pl.DataFrame)
    assert isinstance(is_gpu, bool)
    assert isinstance(dtype, str) and dtype in ["float16", "float32", "float64"]
    assert isinstance(corr_type, str) and corr_type in ["pearson", "spearman", "kendall", "chatterjee"]
    assert isinstance(min_n,      int) and min_n > 0
    assert isinstance(batch_size, int) and batch_size >= 1
    assert isinstance(n_jobs,     int) and n_jobs >= 1
    df_corr = pd.DataFrame(float("nan"), index=df.columns, columns=df.columns)
    batch_size = min(df.shape[1], batch_size)
    if batch_size == 1:
        batch = [np.arange(df.shape[1])]
    else:
        batch = np.array_split(np.arange(df.shape[1]), df.shape[1] // batch_size)
    LOGGER.info("convert to astype...")
    df = df.with_columns(pl.all().cast({"float16": pl.Float32, "float32": pl.Float32, "float64": pl.Float64}[dtype]))
    if corr_type == "spearman":
        assert df.shape[0] <= 100000 # if n data is large, the calculation is explode
        df = df.fill_nan(None).with_columns(pl.all().rank(method='random'))
    elif corr_type == "chatterjee":
        assert df.shape[0] <= 1000000 # if n data is large, the calculation is explode
        df = df.fill_nan(None).with_columns(pl.all().rank(method='random'))
    if is_gpu:
        LOGGER.info(f"calculate correlation [GPU] corr_type: {corr_type}...")
        n_iter = int(((len(batch) ** 2 - len(batch)) / 2) + len(batch))
        i_iter = 0
        for i, batch_x in enumerate(batch):
            for batch_y in batch[i:]:
                i_iter += 1
                LOGGER.info(f"iter: {i_iter} / {n_iter}")
                df_x, df_y = df[:, batch_x], df[:, batch_y]
                input_x, input_y = df_x.to_numpy(), df_y.to_numpy()
                if corr_type == "pearson":
                    ndf_corr = corr_coef_pearson_2array(input_x, input_y, dtype=dtype, min_n=min_n, is_gpu=is_gpu)
                elif corr_type == "spearman":
                    ndf_corr = corr_coef_spearman_2array(df_x, df_y, is_to_rank=False, dtype=dtype, min_n=min_n, is_gpu=is_gpu)
                elif corr_type == "kendall":
                    dictwk = {x:y for x, y in kwargs.items() if x in ["n_sample", "n_iter"]}
                    ndf_corr = corr_coef_kendall_2array(input_x, input_y, dtype=dtype, min_n=min_n, is_gpu=is_gpu, **dictwk)
                elif corr_type == "chatterjee":
                    ndf_corr = corr_coef_chatterjee_2array(df_x, df_y, is_to_rank=False, dtype=dtype, min_n=min_n, is_gpu=is_gpu)
                df_corr.iloc[batch_x, batch_y] = ndf_corr
    else:
        LOGGER.info(f"calculate correlation [CPU] corr_type: {corr_type}...")
        if corr_type == "pearson":
            func1    = partial(corr_coef_pearson_2array_numpy, dtype=getattr(np, dtype), min_n=min_n)
            list_obj = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([
                delayed(lambda x, y, z: (z, func1(x, y)))(df[:, batch_x].to_numpy(), df[:, batch_y].to_numpy(), (batch_x, batch_y))
                for i, batch_x in enumerate(batch) for batch_y in batch[i:]
            ])
        elif corr_type == "spearman":
            func1    = partial(corr_coef_spearman_2array_numpy, is_to_rank=False, dtype=getattr(np, dtype), min_n=min_n)
            list_obj = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([
                delayed(lambda x, y, z: (z, func1(x, y)))(df[:, batch_x].to_numpy(), df[:, batch_y].to_numpy(), (batch_x, batch_y))
                for i, batch_x in enumerate(batch) for batch_y in batch[i:]
            ])
        elif corr_type == "chatterjee":
            func1    = partial(corr_coef_chatterjee_2array, is_to_rank=False, dtype=dtype, min_n=min_n, is_gpu=False)
            list_obj = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([
                delayed(lambda x, y, z: (z, func1(x, y)))(df[:, batch_x], df[:, batch_y], (batch_x, batch_y))
                for i, batch_x in enumerate(batch) for batch_y in batch[i:]
            ])
        else:
            LOGGER.raise_error(f"corr_type: {corr_type} is not supported with CPU")
        for (batch_x, batch_y), ndf_corr in list_obj:
            df_corr.iloc[batch_x, batch_y] = ndf_corr
    LOGGER.info("END")
    return df_corr

def get_sklearn_trees_model_info(model) -> pd.DataFrame:
    LOGGER.info("START")
    df = pd.DataFrame(np.concatenate([x.tree_.feature  for x in model.estimators_]), columns=["i_feature"])
    df["n_sample"]       = np.concatenate([x.tree_.n_node_samples                         for x in model.estimators_])
    df["n_sample_left"]  = np.concatenate([x.tree_.n_node_samples[x.tree_.children_left]  for x in model.estimators_])
    df["n_sample_right"] = np.concatenate([x.tree_.n_node_samples[x.tree_.children_right] for x in model.estimators_])
    df["impurity"]       = np.concatenate([x.tree_.impurity                         for x in model.estimators_])
    df["impurity_left"]  = np.concatenate([x.tree_.impurity[x.tree_.children_left]  for x in model.estimators_])
    df["impurity_right"] = np.concatenate([x.tree_.impurity[x.tree_.children_right] for x in model.estimators_])
    df["importance"]     = df["impurity"] * df["n_sample"] - ((df["impurity_left"] * df["n_sample_left"]) + (df["impurity_right"] * df["n_sample_right"]))
    df = df.loc[df["i_feature"] >= 0]
    LOGGER.info("END")
    return df

def get_features_by_randomtree_importance(
    df: pl.DataFrame, columns_exp: list[str], columns_ans: str, dtype=pl.Float32,
    is_reg: bool=False, max_iter: int=1, min_count: int=100, n_jobs: int=1, 
    **kwargs
) -> pd.DataFrame:
    LOGGER.info("START")
    assert isinstance(df, pl.DataFrame)
    assert isinstance(columns_exp, list) and check_type_list(columns_exp, str)
    assert isinstance(columns_ans, str)
    assert isinstance(is_reg, bool)
    assert dtype in [int, float, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
    assert isinstance(max_iter,   int) and max_iter > 0
    assert isinstance(min_count,  int) and min_count > 0
    # row
    regproc_df = RegistryProc(n_jobs=n_jobs)
    regproc_df.register("ProcDropNa", columns_ans)
    if not is_reg:
        regproc_df.register("ProcCondition", f"{columns_ans} >= 0")
    df = regproc_df.fit(df[columns_exp + [columns_ans]])
    # columns explain
    regproc_exp = RegistryProc(n_jobs=n_jobs)
    regproc_exp.register("ProcAsType", dtype, columns=columns_exp)
    regproc_exp.register("ProcReplaceInf", posinf=float("nan"), neginf=float("nan"))
    regproc_exp.register("ProcToValues")
    regproc_exp.register("ProcFillNaMinMaxRandomly")
    regproc_exp.register("ProcFillNa", 0)
    ndf_x = regproc_exp.fit(df[columns_exp], check_inout=["row"])
    # columns answer
    regproc_ans = RegistryProc(n_jobs=n_jobs)
    if is_reg: regproc_ans.register("ProcAsType", pl.Float64, columns=columns_ans)
    else:      regproc_ans.register("ProcAsType", pl.Int64,   columns=columns_ans)
    regproc_ans.register("ProcToValues")
    regproc_ans.register("ProcReshape", (-1, ))
    ndf_y = regproc_ans.fit(df[[columns_ans]], check_inout=["row"])
    LOGGER.info(f"\ncolumns_exp: {columns_exp[:10]}\ninput:{ndf_x.shape}\ncolumns_ans: {columns_ans}, target:{ndf_y.shape}")
    se_cnt = pd.Series(0, index=np.arange(len(columns_exp)), dtype=float, name="count")
    se_imp = pd.Series(0, index=np.arange(len(columns_exp)), dtype=float, name="importance")
    i = 0
    for i in range(max_iter):
        LOGGER.info(f"create forest. loop: {i}, cnt: {se_cnt.median()}, max: {se_cnt.max()}")
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
        if i == 0: LOGGER.info(f"model: {model}")
        model.fit(ndf_x, ndf_y)
        dfwk    = get_sklearn_trees_model_info(model)
        se_cnt += dfwk.groupby("i_feature").size()
        se_imp += dfwk.groupby("i_feature")["importance"].sum()
        if se_cnt.median() >= min_count: break
    df_feadtures = pd.concat([se_cnt, se_imp], axis=1, ignore_index=False, sort=False)
    df_feadtures.index = columns_exp
    df_feadtures["ratio"] = (df_feadtures["importance"] / df_feadtures["count"])
    LOGGER.info("END")
    return df_feadtures

def get_features_by_adversarial_validation(
    df_train: pl.DataFrame, df_test: pl.DataFrame, columns_exp: list[str], columns_ans: str=None,
    n_split: int=5, n_cv: int=5, dtype=pl.Float32, n_jobs: int=1, **kwargs
):
    LOGGER.info("START")
    assert isinstance(df_train, pl.DataFrame)
    assert isinstance(df_test,  pl.DataFrame)
    assert check_type_list(columns_exp, str)
    assert columns_ans is None or isinstance(columns_ans, str)
    if columns_ans is not None: columns_exp = columns_exp + [columns_ans]
    assert isinstance(n_split, int) and n_split >= 2
    assert isinstance(n_cv,    int) and n_cv    >= 1
    assert dtype in [pl.Float32, pl.Float64]
    # row
    index_df = np.concatenate([np.arange(df_train.shape[0], dtype=int), -1 * (np.arange(df_test.shape[0], dtype=int) + 1)], axis=0)
    # columns explain
    regproc_exp = RegistryProc(n_jobs=n_jobs)
    regproc_exp.register("ProcAsType", dtype, columns=columns_exp)
    regproc_exp.register("ProcToValues")
    regproc_exp.register("ProcReplaceInf", posinf=float("nan"), neginf=float("nan"))
    regproc_exp.register("ProcFillNaMinMaxRandomly")
    regproc_exp.register("ProcFillNa", 0)
    ndf_x1 = regproc_exp.fit(df_train[columns_exp], check_inout=["row"])
    ndf_x2 = regproc_exp(df_test[columns_exp])
    ndf_x  = np.concatenate([ndf_x1, ndf_x2], axis=0)
    # columns answer
    ndf_y = np.concatenate([np.zeros(df_train.shape[0]), np.ones(df_test.shape[0])]).astype(int)
    LOGGER.info(f"input: {ndf_x.shape}, target: {ndf_y.shape}")
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
    LOGGER.info(f"model: {model}")
    # cross validation
    df_imp, df_pred = pd.DataFrame(), pd.DataFrame()
    for i_cv, (index_train, index_test) in enumerate(StratifiedKFold(n_splits=n_split).split(np.arange(ndf_x.shape[0], dtype=int), ndf_y)):
        LOGGER.info(f"cross validation: {i_cv}")
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
    LOGGER.info(f"roc_auc: {se_eval['auc']}, accuracy: {se_eval['acc']}")
    LOGGER.info("END")
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
    LOGGER.info(f"START column: {colname_begin}")
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
    LOGGER.info("END")
    return df_f
