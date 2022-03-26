from typing import List, Union
import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# local package
from kkmlmanager.regproc import RegistryProc
from kkmlmanager.util.dataframe import parallel_apply
from kkmlmanager.util.com import check_type_list
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "get_features_by_variance",
    "corr_coef_gpu",
    "corr_coef_gpu_2array",
    "corr_coef_cpu_2array",
    "get_features_by_correlation",
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
    columns  = df.columns.copy()
    df_sort  = pd.DataFrame(index=np.arange(df.shape[0]))
    list_obj = parallel_apply(df, lambda x: [x.columns, np.sort(x.values, axis=0)], axis=0, batch_size=batch_size, n_jobs=n_jobs)
    df_sort  = pd.concat([pd.DataFrame(y, columns=x) for x, y in list_obj], axis=1, ignore_index=False)
    df_sort  = df_sort.loc[:, columns]
    nsize    = int(df_sort.shape[0] * cutoff) - 1
    if nsize == 0: raise Exception(f"nsize is zero.")
    sebool   = pd.Series(False, index=df_sort.columns)
    if not ignore_nan:
        for i in range(0, df_sort.shape[0] - nsize):
            sewk1  = df_sort.iloc[i        ]
            sewk2  = df_sort.iloc[i + nsize]
            boolwk = ((sewk1.isna() == sewk2.isna()) & sewk1.isna())
            boolwk = boolwk | (sewk1 == sewk2)
            sebool = sebool | boolwk
    else:
        def work(df: pd.DataFrame, cutoff: float):
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
        list_obj = parallel_apply(df_sort, lambda x: work(x, cutoff), axis=0, batch_size=batch_size, n_jobs=n_jobs)
        for x, _ in list_obj: sebool.loc[x.index] = x
    logger.info("END")
    return sebool

def corr_coef_gpu(input: np.ndarray, _dtype=torch.float16, min_n: int=10) -> np.ndarray:
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

def corr_coef_gpu_2array(input_x: np.ndarray, input_y: np.ndarray, _dtype=torch.float16, min_n: int=10) -> np.ndarray:
    logger.info("START")
    assert input_x.shape[0] == input_y.shape[0]
    tens = []
    for input in [input_x, input_y]:
        tensor_max = torch.from_numpy(np.nanmax(input, axis=0)).to(_dtype).to("cuda:0")
        tensor_min = torch.from_numpy(np.nanmin(input, axis=0)).to(_dtype).to("cuda:0")
        tensor_max = (tensor_max - tensor_min)
        tensor_max[tensor_max == 0] = float("inf") # To avoid division by zero
        tens.append(torch.from_numpy(input).to(_dtype).to("cuda:0"))
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
    tens_corr[tens_n_Sxy <= min_n] = float("nan")
    logger.info("END")
    return tens_corr.cpu().numpy()

def corr_coef_cpu_2array(input_x: np.ndarray, input_y: np.ndarray, _dtype=np.float32, min_n: int=10) -> np.ndarray:
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

def get_features_by_correlation(df: pd.DataFrame, cutoff: float=0.9, is_gpu: bool=False, dtype: str="float16", batch_size: int=100, min_n: int=10, n_jobs: int=1):
    logger.info("START")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(cutoff, float) and 0.0 < cutoff <= 1.0
    assert isinstance(is_gpu, bool)
    assert isinstance(min_n,      int) and min_n > 0
    assert isinstance(batch_size, int) and batch_size >= 0
    assert isinstance(n_jobs,     int) and n_jobs >= 1
    df_corr = pd.DataFrame(float("nan"), index=df.columns, columns=df.columns)
    if batch_size == 0:
        batch = [np.arange(df.shape[1])]
    else:
        batch = np.array_split(np.arange(df.shape[1]), df.shape[1] // batch_size)
    if is_gpu:
        for i, batch_x in enumerate(batch):
            for batch_y in batch[i:]:
                input_x, input_y = df.iloc[:, batch_x], df.iloc[:, batch_y]
                input_x, input_y = input_x.values.astype(np.float32), input_y.values.astype(np.float32)
                ndf_corr = corr_coef_gpu_2array(input_x, input_y, _dtype=getattr(torch, dtype), min_n=min_n)
                df_corr.iloc[batch_x, batch_y] = ndf_corr
    else:
        def work(input_x, input_y, dtype, min_n):
            input_x, input_y = input_x.values.astype(np.float32), input_y.values.astype(np.float32)
            ndf_corr = corr_coef_cpu_2array(input_x, input_y, _dtype=getattr(np, dtype), min_n=min_n)
            return ndf_corr
        list_obj = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([
            delayed(lambda x, y, z: (z, work(x, y, dtype, min_n)))(df.iloc[:, batch_x], df.iloc[:, batch_y], (batch_x, batch_y))
            for i, batch_x in enumerate(batch) for batch_y in batch[i:]
        ])
        for (batch_x, batch_y), ndf_corr in list_obj:
            df_corr.iloc[batch_x, batch_y] = ndf_corr
    logger.info("END")
    return df_corr

def get_features_by_randomtree_importance(
    df: pd.DataFrame, columns_exp: List[str], columns_ans: str, 
    is_reg: bool=False, n_estimators: int=100, cnt_thre: int=40, n_jobs: int=1
) -> pd.DataFrame:
    logger.info("START")
    assert isinstance(df, pd.DataFrame)
    assert check_type_list(columns_exp, str)
    assert isinstance(columns_ans, str)
    assert check_type_list(columns_ans, str)
    assert isinstance(is_reg, bool)
    regproc_df = RegistryProc(n_jobs=n_jobs)
    regproc_df.register("ProcDropNa", columns_ans)
    if not is_reg: regproc_df.register("ProcCondition", f"{columns_ans} >= 0")
    df = regproc_df.fit(df[columns_exp + [columns_ans]])
    regproc_exp = RegistryProc(n_jobs=n_jobs)
    regproc_exp.register("ProcAsType", np.float32, batch_size=10)
    regproc_exp.register("ProcToValues")
    regproc_exp.register("ProcFillNaMinMax")
    regproc_exp.register("ProcFillNa", 0)
    ndf_x = regproc_exp.fit(df[columns_exp])
    regproc_ans = RegistryProc(n_jobs=1)
    if is_reg: regproc_ans.register("ProcAsType", np.float32)
    else:      regproc_ans.register("ProcAsType", np.int32)
    regproc_ans.register("ProcReshape", (-1, ))
    ndf_y = regproc_exp.fit(df[[columns_ans]])
    logger.info(f"columns_exp: {columns_exp}\ninput:{ndf_x.shape}\ncolumns_ans: {columns_ans}, answer:{ndf_y.shape}")
    df_features_cnt = pd.DataFrame(columns=columns_exp)
    df_features_imp = pd.DataFrame(columns=columns_exp)
    i = 0
    while(True):
        i += 1
        logger.info("create forest. loop:%s", i)
        # モデルの定義(木の数はとりあえず100固定)
        model = None
        dictwk = {"bootstrap":False, "n_estimators":n_estimators, "max_depth":10, "max_features":"auto", "verbose":3, "random_state":i, "n_jobs": n_jobs}
        if is_reg: model = ExtraTreesRegressor(**dictwk)
        else:      model = ExtraTreesClassifier(**dictwk)
        model.fit(ndf_x, ndf_y)
        ## model内で特徴量を使用した回数をカウントする
        feature_count = np.hstack(list(map(lambda y: y.tree_.feature, model.estimators_)))
        feature_count = feature_count[feature_count >= 0] #-1以下は子がないとか特別の意味を持つ
        sewk = pd.DataFrame(colname_explain[feature_count], columns=[0]).groupby(0).size()
        df_features_cnt = df_features_cnt.append(sewk, ignore_index=True)

        ## modelの重要度を格納する
        sewk = pd.Series(model.feature_importances_, index=df_features_imp.columns.values.copy())
        df_features_imp = df_features_imp.append(sewk, ignore_index=True)

        logger.debug("\n%s", df_features_cnt)
        logger.debug("\n%s", df_features_imp)

        ## カウントが一定数達した場合に終了する
        ## ※例えば、ほとんどがnanのデータは分岐できる点が少ないためカウントが少なくなる
        cnt = df_features_cnt.sum(axis=0).median() #各特長量毎の合計の中央値
        logger.info("count median:%s", cnt)
        if cnt >= cnt_thre:
            break

    # 特徴量計算
    ## カウントがnanの箇所は、重要度もnanで置き換える(変換はndfを通して参照形式で行う)
    ## カウントが無ければnan. 重要度は、カウントがなければ0
    ndf_cnt = df_features_cnt.values.astype(np.float32)
    ndf_imp = df_features_imp.values.astype(np.float32)
    ndf_imp[np.isnan(ndf_cnt)] = np.nan #木の分岐にはあるが重要度として0がカウントする可能性も考慮して
    ## 重要度の計算
    df_features = df_features_imp.mean().reset_index().copy()
    df_features.columns  = ["feature_name","importance"]
    df_features["std"]   = df_features_imp.std().values #カラム順を変えていないので、joinしなくても良いはず
    df_features["count"] = df_features_cnt.sum().values
    df_features = df_features.sort_values("importance", ascending=False)

    logger.info("END")
    return df_features
