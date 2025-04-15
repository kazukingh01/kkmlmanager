import numpy as np
from functools import partial
from joblib import Parallel, delayed


__all__ = [
    "isin_compare_string",
    "parallel_apply",
    "NdarrayWithErr",
    "nperr_stack",
    "nperr_concat",
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
    func1 = partial(parallel_apply_func1, func=func)
    if axis == 0:
        batch = np.arange(ndf.shape[1])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(func1)(i_batch, ndf[:, i_batch]) for i_batch in batch])
    else:
        batch = np.arange(ndf.shape[0])
        if batch_size > 1: batch = np.array_split(batch, batch.shape[0] // batch_size)
        else: batch = batch.reshape(-1, 1)
        list_object = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size="auto")([delayed(func1)(i_batch, ndf[i_batch   ]) for i_batch in batch])
    indexes = np.concatenate([x for x, y in list_object])
    if axis == 0: output = np.concatenate([y for x, y in list_object], axis=1)[:, np.argsort(indexes)]
    else:         output = np.concatenate([y for x, y in list_object], axis=0)[np.argsort(indexes), :]
    return output
def parallel_apply_func1(x, y, func=None):
    return [x, func(y)]

class NdarrayWithErr:
    def __init__(self, val: np.ndarray | int | float, err: np.ndarray | int | float):
        if isinstance(val, (int, float)):
            assert isinstance(err, (int, float))
            val = np.array([val])
            err = np.array([err])
        else:
            assert isinstance(val, np.ndarray)
            assert isinstance(err, np.ndarray)
        assert val.shape == err.shape
        self.val = val
        self.err = err
    def __getitem__(self, idx):
        return __class__(self.val[idx], self.err[idx])
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return __class__(self.val + other.val, np.sqrt(self.err ** 2 + other.err ** 2))
        else:
            return __class__(self.val + other, self.err)
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return __class__(self.val - other.val, np.sqrt(self.err ** 2 + other.err ** 2))
        else:
            return __class__(self.val - other, self.err)
    def __rsub__(self, other):
        return self.__sub__(other)
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return __class__(self.val * other.val, np.sqrt((other.val * self.err) ** 2 + (self.val * other.err) ** 2))
        else:
            return __class__(self.val * other, self.err * other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return __class__(self.val / other.val, np.sqrt((self.err / other.val) ** 2 + (other.err * self.val / (other.val ** 2)) ** 2))
        else:
            return __class__(self.val / other, self.err / other)
    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return __class__(other.val / self.val, np.sqrt((other.err / self.val) ** 2 + (self.err * other.val / (self.val ** 2)) ** 2))
        else:
            return __class__(other / self.val, np.abs(other / (self.val ** 2)) * self.err)
    def __pos__(self):
        return self
    def __neg__(self):
        return __class__(-self.val, self.err)
    def __abs__(self):
        return __class__(np.abs(self.val), self.err)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.val == other.val) & (self.err == other.err)
        else:
            return (self.val == other)
    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return (self.val != other.val) | (self.err != other.err)
        else:
            return (self.val != other)
    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.val > other.val
        else:
            return self.val > other
    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return self.val >= other.val
        else:
            return self.val >= other
    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.val < other.val
        else:
            return self.val < other
    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self.val <= other.val
        else:
            return self.val <= other
    def __pow__(self, other: int | float):
        if isinstance(other, self.__class__):
            raise ValueError("Cannot calculate statistical error.")
        else:
            return __class__(self.val ** other, np.abs(other * (self.val ** (other - 1))) * self.err)
    def __str__(self):
        return f"val: {self.val.__str__()}, err: {self.err.__str__()}"
    def __repr__(self):
        return f"val: {self.val.__repr__()}, err: {self.err.__repr__()}"
    @property
    def shape(self):
        return self.val.shape
    @property
    def size(self):
        return self.val.size
    @property
    def T(self):
        return __class__(self.val.T, self.err.T)
    def sum(self, *args, **kwargs):
        return __class__(self.val.sum(*args, **kwargs), np.sqrt((self.err ** 2).sum(*args, **kwargs)))
    def mean(self, *args, **kwargs):
        val = self.val.mean(*args, **kwargs)
        var = np.sqrt((self.err ** 2).sum(*args, **kwargs)) / (self.val.size / val.size)
        return __class__(val, var)
    def median(self, *args, **kwargs):
        """
        https://en.wikipedia.org/wiki/Median#Medians_for_samples
        "the relative standard error of the median will be (𝜋/2)^0.5 ~ 1.25, or 25% greater than the standard error of the mean"
        """
        val = np.median(self.val, *args, **kwargs)
        n   = self.val.size // val.size
        err = self.mean(*args, **kwargs).err
        return __class__(val, np.sqrt(np.pi / (2.0 * n)) * err)
    def _minmax(self, *args, _proc: str=None, **kwargs):
        assert _proc in ["argmin", "argmax"]
        val   = getattr(self.val, _proc)(*args, **(kwargs | {"keepdims": True}))
        shape = getattr(self.val, _proc)(*args, **kwargs).shape
        idx   = np.arange(self.val.size, dtype=int).reshape(self.val.shape)
        if val.size == 1:
            idx = val.reshape(-1)
        else:
            boolwk = True
            for i, (x, n) in enumerate(zip(val.shape, self.val.shape)):
                if x != n:
                    valwk  = np.moveaxis(val, i, 0)
                    idxwk  = np.moveaxis(idx, i, 0)
                    idx    = idxwk.reshape(n, -1)[valwk.reshape(-1), np.arange(valwk.size, dtype=int)]
                    boolwk = False
                    break
            if boolwk:
                raise ValueError("This is unexpected case.")
        val = self.val.reshape(-1)[idx].reshape(shape)
        err = self.err.reshape(-1)[idx].reshape(shape)
        return __class__(val, err)
    def min(self, *args, **kwargs):
        return self._minmax(*args, _proc="argmin", **kwargs)
    def max(self, *args, **kwargs):
        return self._minmax(*args, _proc="argmax", **kwargs)
    def reshape(self, *args, **kwargs):
        return __class__(self.val.reshape(*args, **kwargs), self.err.reshape(*args, **kwargs))
    def to_numpy(self, axis_normalize: int | None=None):
        if axis_normalize is None:
            return self.val
        else:
            weight = 1.0 / np.clip(self.err / self.val, 1e-20, 1.0)
            val    = (self.val / self.val.sum(axis=axis_normalize, keepdims=True)) * weight
            return val / val.sum(axis=axis_normalize, keepdims=True)

def nperr_stack(list_ndferr: list[NdarrayWithErr], *args, **kwargs) -> NdarrayWithErr:
    val = np.stack([x.val for x in list_ndferr], *args, **kwargs)
    err = np.stack([x.err for x in list_ndferr], *args, **kwargs)
    return NdarrayWithErr(val, err)

def nperr_concat(list_ndferr: list[NdarrayWithErr], *args, **kwargs) -> NdarrayWithErr:
    val = np.concatenate([x.val for x in list_ndferr], *args, **kwargs)
    err = np.concatenate([x.err for x in list_ndferr], *args, **kwargs)
    return NdarrayWithErr(val, err)

