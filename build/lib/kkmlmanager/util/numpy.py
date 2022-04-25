import numpy as np
from joblib import Parallel, delayed


__all__ = [
    "isin_compare_string",
]


def isin_compare_string(input: np.ndarray, target: np.ndarray):
    if len(input)  == 0: return np.zeros(0, dtype=bool)
    if len(target) == 0: return np.zeros_like(input, dtype=bool)
    dictwk = {x:i for i, x in enumerate(target)}
    col_target = np.vectorize(lambda x: dictwk.get(x) if x in dictwk else -1)(target).astype(int)
    col_input  = np.vectorize(lambda x: dictwk.get(x) if x in dictwk else -1)(input)
    col_input  = col_input.astype(int)
    return np.isin(col_input, col_target)

def parallel_apply(ndf: np.ndarray, func, axis: int=0, batch_size: int=1, n_jobs: int=1):
    assert isinstance(ndf, np.ndarray)
    assert isinstance(axis, int) and axis in [0, 1]
    assert isinstance(batch_size, int) and batch_size >= 1
    assert isinstance(n_jobs, int) and n_jobs > 0
    if axis == 0: batch_size = min(ndf.shape[1], batch_size)
    else:         batch_size = min(ndf.shape[0], batch_size)
    list_object = None
    if axis == 0:
        batch = np.arange(ndf.shape[1])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(lambda x, y: [x, func(y)])(i_batch, ndf[:, i_batch]) for i_batch in batch])
    else:
        batch = np.arange(ndf.shape[0])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(lambda x, y: [x, func(y)])(i_batch, ndf[i_batch   ]) for i_batch in batch])
    indexes = np.concatenate([x for x, y in list_object])
    if axis == 0: output = np.concatenate([y for x, y in list_object], axis=1)[:, np.argsort(indexes)]
    else:         output = np.concatenate([y for x, y in list_object], axis=0)[np.argsort(indexes), :]
    return output
