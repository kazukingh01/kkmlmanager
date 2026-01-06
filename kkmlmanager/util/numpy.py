import typing
import numpy as np
from scipy.stats import norm
from functools import partial
from joblib import Parallel, delayed
from sklearn.isotonic import isotonic_regression


__all__ = [
    "isin_compare_string",
    "parallel_apply",
    "NdarrayWithErr",
    "stack",
    "concatenate",
    "take_along_axis",
    "isotonic_regression_with_err",
    "normalize",
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
    def __array__(self, dtype=None):
        return self.val.__array__(dtype=dtype)
    def __getitem__(self, idx) -> typing.Self:
        return __class__(self.val[idx], self.err[idx])
    def __add__(self, other) -> typing.Self:
        if isinstance(other, self.__class__):
            return __class__(self.val + other.val, np.sqrt(self.err ** 2 + other.err ** 2))
        else:
            return __class__(self.val + other, self.err)
    def __radd__(self, other) -> typing.Self:
        return self.__add__(other)
    def __sub__(self, other) -> typing.Self:
        if isinstance(other, self.__class__):
            return __class__(self.val - other.val, np.sqrt(self.err ** 2 + other.err ** 2))
        else:
            return __class__(self.val - other, self.err)
    def __rsub__(self, other) -> typing.Self:
        return self.__sub__(other)
    def __mul__(self, other) -> typing.Self:
        if isinstance(other, self.__class__):
            return __class__(self.val * other.val, np.sqrt((other.val * self.err) ** 2 + (self.val * other.err) ** 2))
        else:
            return __class__(self.val * other, self.err * other)
    def __rmul__(self, other) -> typing.Self:
        return self.__mul__(other)
    def __truediv__(self, other) -> typing.Self:
        if isinstance(other, self.__class__):
            return __class__(self.val / other.val, np.sqrt((self.err / other.val) ** 2 + (other.err * self.val / (other.val ** 2)) ** 2))
        else:
            return __class__(self.val / other, self.err / other)
    def __rtruediv__(self, other) -> typing.Self:
        if isinstance(other, self.__class__):
            return __class__(other.val / self.val, np.sqrt((other.err / self.val) ** 2 + (self.err * other.val / (self.val ** 2)) ** 2))
        else:
            return __class__(other / self.val, np.abs(other / (self.val ** 2)) * self.err)
    def __pos__(self) -> typing.Self:
        return self
    def __neg__(self) -> typing.Self:
        return __class__(-self.val, self.err)
    def __abs__(self) -> typing.Self:
        return __class__(np.abs(self.val), self.err)
    def __eq__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return (self.val == other.val) & (self.err == other.err)
        else:
            return (self.val == other)
    def __ne__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return (self.val != other.val) | (self.err != other.err)
        else:
            return (self.val != other)
    def __gt__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return self.val > other.val
        else:
            return self.val > other
    def __ge__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return self.val >= other.val
        else:
            return self.val >= other
    def __lt__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return self.val < other.val
        else:
            return self.val < other
    def __le__(self, other) -> np.ndarray:
        if isinstance(other, self.__class__):
            return self.val <= other.val
        else:
            return self.val <= other
    def __pow__(self, other: int | float) -> typing.Self:
        if isinstance(other, self.__class__):
            raise ValueError("Cannot calculate statistical error.")
        else:
            return __class__(self.val ** other, np.abs(other * (self.val ** (other - 1))) * self.err)
    def __str__(self) -> str:
        return f"val: {self.val.__str__()}, err: {self.err.__str__()}"
    def __repr__(self) -> str:
        return f"val: {self.val.__repr__()}, err: {self.err.__repr__()}"
    @property
    def shape(self) -> tuple[int, ...]:
        return self.val.shape
    @property
    def size(self) -> int:
        return self.val.size
    @property
    def ndim(self) -> int:
        return self.val.ndim
    @property
    def T(self) -> typing.Self:
        return __class__(self.val.T, self.err.T)
    def sum(self, *args, **kwargs) -> typing.Self:
        return __class__(self.val.sum(*args, **kwargs), np.sqrt((self.err ** 2).sum(*args, **kwargs)))
    def mean(self, *args, **kwargs) -> typing.Self:
        val = self.val.mean(*args, **kwargs)
        var = np.sqrt((self.err ** 2).sum(*args, **kwargs)) / (self.val.size / val.size)
        return __class__(val, var)
    def median(self, *args, **kwargs) -> typing.Self:
        """
        https://en.wikipedia.org/wiki/Median#Medians_for_samples
        "the relative standard error of the median will be (𝜋/2)^0.5 ~ 1.25, or 25% greater than the standard error of the mean"
        """
        val = np.median(self.val, *args, **kwargs)
        n   = self.val.size // val.size
        err = self.mean(*args, **kwargs).err
        return __class__(val, np.sqrt(np.pi / (2.0 * n)) * err)
    def _minmax(self, *args, _proc: str=None, **kwargs) -> typing.Self:
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
    def min(self, *args, **kwargs) -> typing.Self:
        return self._minmax(*args, _proc="argmin", **kwargs)
    def max(self, *args, **kwargs) -> typing.Self:
        return self._minmax(*args, _proc="argmax", **kwargs)
    def reshape(self, *args, **kwargs) -> typing.Self:
        return __class__(self.val.reshape(*args, **kwargs), self.err.reshape(*args, **kwargs))
    def to_numpy(self) -> np.ndarray:
        return self.val
    def copy(self) -> typing.Self:
        return __class__(self.val.copy(), self.err.copy())

def stack(list_ndferr: list[np.ndarray | NdarrayWithErr], *args, **kwargs) -> np.ndarray | NdarrayWithErr:
    assert isinstance(list_ndferr, list)
    if all(isinstance(x, np.ndarray) for x in list_ndferr):
        return np.stack(list_ndferr, *args, **kwargs)
    else:
        assert all(isinstance(x, NdarrayWithErr) for x in list_ndferr)
        val = np.stack([x.val for x in list_ndferr], *args, **kwargs)
        err = np.stack([x.err for x in list_ndferr], *args, **kwargs)
        return NdarrayWithErr(val, err)

def concatenate(list_ndferr: list[np.ndarray | NdarrayWithErr], *args, **kwargs) -> np.ndarray | NdarrayWithErr:
    assert isinstance(list_ndferr, list)
    if all(isinstance(x, np.ndarray) for x in list_ndferr):
        return np.concatenate(list_ndferr, *args, **kwargs)
    else:
        assert all(isinstance(x, NdarrayWithErr) for x in list_ndferr)
        val = np.concatenate([x.val for x in list_ndferr], *args, **kwargs)
        err = np.concatenate([x.err for x in list_ndferr], *args, **kwargs)
        return NdarrayWithErr(val, err)

def take_along_axis(arr: np.ndarray | NdarrayWithErr, *args, **kwargs):
    assert isinstance(arr, (np.ndarray, NdarrayWithErr))
    if isinstance(arr, np.ndarray):
        return np.take_along_axis(arr, *args, **kwargs)
    else:
        return NdarrayWithErr(np.take_along_axis(arr.val, *args, **kwargs), np.take_along_axis(arr.err, *args, **kwargs))

def isotonic_regression_with_err(input_x: NdarrayWithErr, y_min: float=0.0, y_max: float=np.inf, n_bins: int=10) -> np.ndarray:
    """
    >>> isotonic_regression([0,1,2,2,1,3], sample_weight=[1,1,1,1,1,1], y_min=0, y_max=5)
    array([0.        , 1.        , 1.66666667, 1.66666667, 1.66666667,
        3.        ])
    >>> isotonic_regression([0,1,2,2,1,3], sample_weight=[1,1,9,1,1,1], y_min=0, y_max=5)
    array([0.        , 1.        , 1.90909091, 1.90909091, 1.90909091,
        3.        ])
    >>> isotonic_regression([0,1,2,2,1,3], sample_weight=[1,1,1,1,1,1], y_min=0, y_max=1)
    array([0., 1., 1., 1., 1., 1.])
    """
    assert isinstance(input_x, NdarrayWithErr)
    assert input_x.ndim == 2
    assert input_x.shape[1] >= 2
    assert isinstance(n_bins, int) and n_bins >= 10
    input_x.err = np.clip(input_x.err, min=1e-10, max=y_max)
    idx      = np.argsort(input_x.val, axis=-1)
    sorted_x = take_along_axis(input_x, idx, axis=-1)
    weight   = 1.0 / (sorted_x.err ** 2)
    alpha    = norm.cdf(-2.0) # -2 sigma
    beta     = norm.cdf(+2.0) # +2 sigma
    x_p      = norm.ppf(alpha + (np.arange(1, n_bins + 1) - 0.5) * (beta - alpha) / n_bins)
    x_p      = x_p.reshape(*([1,]*input_x.ndim + [-1,]))
    reshaped = sorted_x.reshape(*(list(sorted_x.shape) + [1,]))
    addnoise = reshaped.val + reshaped.err * x_p
    addnoise = np.clip(addnoise, min=y_min, max=y_max)
    addnoise = np.moveaxis(addnoise, 2, 1)
    weight   = np.moveaxis(weight.repeat(n_bins).reshape(-1, weight.shape[1], n_bins), 2, 1)
    ndfbool  = np.stack([addnoise[::, i] > addnoise[::, i + 1] for i in range(addnoise.shape[-1] - 1)]).sum(axis=0).sum(axis=-1).astype(bool)
    if ndfbool.sum() == 0:
        return input_x.val
    else:
        ndfret   = input_x.val.copy()
        idxrow   = np.arange(input_x.shape[0])[ndfbool]
        idx      = idx[ndfbool]
        addnoise = addnoise[ndfbool]
        weight   = weight[ndfbool]
        vals     = [
            isotonic_regression(x, sample_weight=w, y_min=y_min, y_max=y_max)
            for x, w in zip(addnoise.reshape(-1, input_x.shape[1]), weight.reshape(-1, input_x.shape[1]))
        ]
        vals     = np.stack(vals).reshape(-1, n_bins, input_x.shape[1]).mean(axis=-2)
        for i, (_idxrow, _idx) in enumerate(zip(idxrow, idx)):
            ndfret[_idxrow, _idx] = vals[i]
        return ndfret

def normalize(input_x: np.ndarray | NdarrayWithErr, y_min: float=0.0, y_max: float=np.inf, n_bins: int=100) -> np.ndarray:
    assert isinstance(input_x, (np.ndarray, NdarrayWithErr))
    assert input_x.ndim == 2
    if isinstance(input_x, NdarrayWithErr):
        output = isotonic_regression_with_err(input_x, y_min=y_min, y_max=y_max, n_bins=n_bins)
    else:
        output = input_x
    return output / output.sum(axis=-1, keepdims=True)

def object_nptype_to_pytype(_o: list | tuple | dict | str | int | float | bool) -> typing.Any:
    if isinstance(_o, list):
        return [object_nptype_to_pytype(x) for x in _o]
    elif isinstance(_o, tuple):
        return tuple([object_nptype_to_pytype(x) for x in _o])
    elif isinstance(_o, dict):
        return {object_nptype_to_pytype(x): object_nptype_to_pytype(y) for x, y in _o.items()}
    elif isinstance(_o, np.ndarray):
        return object_nptype_to_pytype(_o.tolist())
    elif isinstance(_o, (np.str_, str)):
        return str(_o)
    elif isinstance(_o, (int, np.int64, np.int32, np.int16, np.int8)):
        return int(_o)
    elif isinstance(_o, (float, np.float64, np.float32, np.float16)):
        return float(_o)
    elif isinstance(_o, (bool, np.bool)):
        return bool(_o)
    else:
        assert False, f"Unexpected type: {type(_o)}"
