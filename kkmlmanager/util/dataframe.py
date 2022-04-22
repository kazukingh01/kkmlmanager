import re
import pandas as pd
import numpy as np
from typing import List
from joblib import Parallel, delayed

# local package
from kkmlmanager.util.com import check_type_list


__all__ = [
    "parallel_apply",
    "astype_faster",
    "query",
]


def parallel_apply(df: pd.DataFrame, func, axis: int=0, group_key=None, func_aft=None, batch_size: int=1, n_jobs: int=1):
    """
    pandarallel is slow in some cases. It is twice as fast to use pandas.
    Params::
        func:
            ex) lambda x: x.rank()
        axis:
            axis=0: df.apply(..., axis=0)
            axis=1: df.apply(..., axis=1)
            axis=2: df.groupby(...)
        func_aft:
            input: (list_object, index, columns)
            ex) lambda x,y,z: pd.concat(x, axis=1, ignore_index=False, sort=False).loc[:, z]
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(axis, int) and axis in [0, 1, 2]
    if axis == 2: assert group_key is not None and check_type_list(group_key, str)
    assert isinstance(batch_size, int) and batch_size >= 1
    assert isinstance(n_jobs, int) and n_jobs > 0
    if   axis == 0: batch_size = min(df.shape[1], batch_size)
    elif axis == 1: batch_size = min(df.shape[0], batch_size)
    index, columns = df.index, df.columns
    list_object = None
    if   axis == 0:
        batch = np.arange(df.shape[1])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(func)(df.iloc[:, i_batch]) for i_batch in batch])
    elif axis == 1:
        batch = np.arange(df.shape[0])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(func)(df.iloc[i_batch   ]) for i_batch in batch])
    else:
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size=batch_size)([delayed(func)(dfwk) for _, dfwk in df.groupby(group_key)])
    if len(list_object) > 0 and func_aft is not None:
        return func_aft(list_object, index, columns)
    else:
        return list_object

def astype_faster(df: pd.DataFrame, list_astype: List[dict]=[], batch_size: int=1, n_jobs: int=1):
    """
    list_astype:
        [{"from": from_dtype, "to": to_dtype}, {...}]
        ex) [{"from": np.float64, "to": np.float16}, {"from": ["aaaa", "bbbb"], "to": np.float16}]
    """
    assert isinstance(n_jobs, int) and n_jobs >= 1
    assert check_type_list(list_astype, dict) and len(list_astype) > 0
    assert isinstance(batch_size, int) and batch_size >= 1
    df      = df.copy()
    columns = df.columns.copy()
    for dictwk in list_astype:
        from_dtype, to_dtype = dictwk["from"], dictwk["to"]
        colbool = None
        if from_dtype is None or from_dtype == slice(None):
            colbool = np.ones(df.shape[1]).astype(bool)
        elif isinstance(from_dtype, type):
            colbool = (df.dtypes == from_dtype)
        elif isinstance(from_dtype, np.ndarray):
            assert len(from_dtype.shape) == 1
            if from_dtype.shape[0] == 0: continue
            if from_dtype.dtype == bool:
                assert from_dtype.shape[0] == df.columns.shape[0]
                colbool = from_dtype
            elif from_dtype.dtype in [str, object]:
                colbool = df.columns.isin(from_dtype)
        if colbool is None:
            raise Exception(f"from_dtype: {from_dtype} is not matched.")
        colbool = ((df.dtypes != to_dtype).values & colbool)
        if colbool.sum() > 0:
            dfwk = parallel_apply(
                df.loc[:, colbool].copy(), lambda x: x.astype(to_dtype), axis=0, 
                func_aft=lambda x,y,z: pd.concat(x, axis=1, ignore_index=False, sort=False), 
                batch_size=batch_size, n_jobs=n_jobs
            )
            df = df.loc[:, ~colbool]
            df = pd.concat([df, dfwk], axis=1, ignore_index=False, sort=False)
    df = df.loc[:, columns]
    return df

def query(df: pd.DataFrame, str_where: str):
    assert isinstance(str_where, str) and str_where.find("(") < 0 and str_where.find(")") < 0
    str_where = [x.strip() for x in re.split("(and|or)", str_where)]
    list_bool = []
    for phrase in str_where:
        if phrase in ["and", "or"]:
            list_bool.append(phrase)
        else:
            colname, operator, value = [x.strip() for x in re.split("( = | > | < | >= | <= | in )", phrase)]
            if len(re.findall("^\[.+\]$", value)) > 0:
                value = [x.strip() for x in value[1:-1].split(",")]
                value = [int(x) if x.find("'") < 0 else x for x in value]
            elif value.find("'") < 0:
                value = int(value)
            ndf_bool = None
            if   operator == "=":  ndf_bool = (df[colname] == value).values
            elif operator == ">":  ndf_bool = (df[colname] >  value).values 
            elif operator == "<":  ndf_bool = (df[colname] <  value).values 
            elif operator == ">=": ndf_bool = (df[colname] >= value).values 
            elif operator == "<=": ndf_bool = (df[colname] <= value).values 
            elif operator == "in": ndf_bool = (df[colname].isin(value)).values
            else: raise ValueError(f"'{operator}' is not supported.")
            list_bool.append(ndf_bool.copy())
    ndf_bool, operator = None, 0
    for x in list_bool:
        if isinstance(x, np.ndarray):
            if ndf_bool is None:
                ndf_bool = x
            else:
                if operator == 0:
                    ndf_bool = (ndf_bool & x)
                else:
                    ndf_bool = (ndf_bool | x)
        elif x == "and": operator = 0
        elif x == "or":  operator = 1
    return ndf_bool