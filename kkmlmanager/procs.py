import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.decomposition import PCA
try:
    # The import time is vely slow, which is about 8 second. So this import module is optional.
    import umap # pip install umap-learn==0.5.5
except ModuleNotFoundError:
    pass

# local package
from kkmlmanager.util.dataframe import query
from kkmlmanager.util.com import check_type, check_type_list
from kklogger import set_logger
logger = set_logger(__name__)


__all__ = [
    "info_columns",
    "NotFittedError",
    "BaseProc",
    "ProcMinMaxScaler",
    "ProcStandardScaler",
    "ProcRankGauss",
    "ProcPCA",
    "ProcOneHotEncoder",
    "ProcFillNa",
    "ProcFillNaMinMaxRandomly",
    "ProcReplaceValue",
    "ProcReplaceInf",
    "ProcToValues",
    "ProcMap",
    "ProcAsType",
    "ProcReshape",
    "ProcDropNa",
    "ProcCondition",
    "ProcDigitize",
    "ProcEval",
]


def info_columns(input: pd.DataFrame | np.ndarray | pl.DataFrame) -> pd.Index | list | tuple:
    assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
    if isinstance(input, pd.DataFrame):
        assert input.columns.dtype == object
        shape: pd.Index = input.columns.copy()
    elif isinstance(input, pl.DataFrame):
        shape: list = input.columns.copy()
    else:
        shape: tuple = input.shape[1:]
    return shape

class NotFittedError(Exception):
    pass

class BaseProc:
    def __init__(self, n_jobs: int=1, is_jobs_fix: bool=False):
        assert isinstance(n_jobs, int) and n_jobs >= 1
        assert isinstance(is_jobs_fix, bool)
        self.is_check    = False
        self.is_fit      = False
        self.type_in     = None
        self.type_out    = None
        self.n_jobs      = n_jobs
        self.is_jobs_fix = is_jobs_fix
    def fit(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, *args, **kwargs):
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        self.type_in  = {pd.DataFrame: "pd", np.ndarray: "np", pl.DataFrame: "pl"}[type(input)]
        self.shape_in = info_columns(input)
        output        = self.fit_main(input, *args, **kwargs)
        self.is_fit   = True
        if output is None:
            output = self(input, *args, is_check=False, **kwargs)
        self.type_out  = {pd.DataFrame: "pd", np.ndarray: "np", pl.DataFrame: "pl"}[type(output)]
        self.shape_out = info_columns(input)
        return output
    def __call__(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, *args, n_jobs: int=None, is_check: bool=None, **kwargs):
        if is_check is not None:
            assert isinstance(is_check, bool)
        else:
            is_check = self.is_check
        if not self.is_fit:
            raise NotFittedError("You must use 'fit' first.")
        if self.is_jobs_fix == False and n_jobs is not None:
            assert isinstance(n_jobs, int) and n_jobs >= -1
            self.n_jobs = n_jobs
        if is_check:
            if self.type_in == "pd":
                assert len(self.shape_in) == len(input.columns)
                assert np.all(self.shape_in == input.columns)
            elif self.type_in == "pl":
                assert self.shape_in == input.columns
            else:
                assert self.shape_in == input.shape[1:]
        output = self.call_main(input, *args, **kwargs)
        if is_check:
            if self.type_out == "pd":
                assert len(self.shape_out) == len(output.columns)
                assert np.all(self.shape_out == output.columns)
            elif self.type_out == "pl":
                assert self.shape_out == output.columns
            else:
                assert self.shape_out == output.shape[1:]
        return output
    def __str__(self):
        attrs_str = ', '.join(f'{k}={v!r}' for k, v in vars(self).items())
        return f'{self.__class__.__name__}({attrs_str})'
    def fit_main(self):
        raise NotImplementedError()
    def call_main(self):
        raise NotImplementedError()

class ProcSKLearn(BaseProc):
    def __init__(self, class_proc, *args, **kwargs):
        assert isinstance(class_proc, type)
        super().__init__()
        self.proc = class_proc(*args, **kwargs)
    def __str__(self): return str(self.proc)
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if isinstance(input, pl.DataFrame):
            input = input.to_pandas()
        return self.proc.fit_transform(input)
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if isinstance(input, pl.DataFrame):
            input = input.to_pandas()
        return self.proc.transform(input)

class ProcMinMaxScaler(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(MinMaxScaler, *args, **kwargs)

class ProcStandardScaler(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(StandardScaler, *args, **kwargs)

class ProcRankGauss(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(QuantileTransformer, *args, output_distribution="normal", **kwargs)

class ProcPCA(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(PCA, *args, **kwargs)

class ProcOneHotEncoder(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(OneHotEncoder, *args, **kwargs)
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        input = super().fit_main(input)
        return input.toarray()
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        input = super().call_main(input)
        return input.toarray()

class ProcUMAP(ProcSKLearn):
    """
    see: https://github.com/lmcinnes/umap
    """
    def __init__(self, *args, **kwargs):
        super().__init__(umap.UMAP, *args, **kwargs)

class ProcFillNa(BaseProc):
    def __init__(self, fill_value: str | int | float | list | np.ndarray, **kwargs):
        assert check_type(fill_value, [str, int, float, list, np.ndarray])
        if isinstance(fill_value, str):
            assert fill_value in ["mean", "max", "min", "median"]
        super().__init__(**kwargs)
        self.fill_value = fill_value
        self.fit_values = None
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in == "pd":
            if isinstance(self.fill_value, str):
                self.fit_values = {x: y for x, y in getattr(input, self.fill_value)(axis=0).items()}
        elif self.type_in == "pl":
            if isinstance(self.fill_value, str):
                self.fit_values = getattr(input, self.fill_value)().to_dicts()[0]
        else:
            assert len(input.shape) == 2
            if isinstance(self.fill_value, str):
                self.fit_values = getattr(np, f"nan{self.fill_value}")(input, axis=0)
        if isinstance(self.fill_value, list):
            assert check_type_list(self.fill_value, [int, float])
            assert len(self.fill_value) == input.shape[-1]
            self.fit_values = np.array(self.fill_value)
        elif isinstance(self.fill_value, np.ndarray):
            assert self.fill_value.shape[0] == input.shape[-1]
            self.fit_values = self.fill_value
        else:
            self.fit_values = self.fill_value
        assert check_type(self.fit_values, [int, float, np.ndarray, dict])
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        output = input # Don't use copy()
        if self.type_in == "pd":
            if isinstance(self.fit_values, np.ndarray):
                output = output.fillna({x: y for x, y in zip(output.columns, self.fit_values)})
            else:
                output = output.fillna(self.fit_values) # same as dict
        elif self.type_in == "pl":
            if isinstance(self.fit_values, np.ndarray):
                assert len(output.columns) == self.fit_values.shape[0]
                output = output.with_columns([pl.col(x).fill_nan(None).fill_null(y) for x, y in zip(output.columns, self.fit_values)])
            elif isinstance(self.fit_values, dict):
                output = output.with_columns([pl.col(x).fill_nan(None).fill_null(y) for x, y in self.fit_values.items()])
            else:
                output = output.fill_nan(None).fill_null(self.fit_values)
        else:
            if isinstance(self.fit_values, np.ndarray):
                mask = np.isnan(output).copy()
                for i, x in enumerate(mask.T):
                    output[x, i] = self.fit_values[i]
            elif isinstance(self.fit_values, dict):
                raise ValueError(f"type_in: {self.type_in}, fit_values: {self.fit_values} is not supported")
            else:
                output = np.nan_to_num(input, nan=self.fit_values)
            pass
        return output

class ProcFillNaMinMaxRandomly(BaseProc):
    def __init__(self, add_value: int | float=1.0, **kwargs):
        assert check_type(add_value, [int, float])
        super().__init__(**kwargs)
        self.add_value  = add_value
        self.fit_values = None
    def fit_main(self, input: np.ndarray):
        assert isinstance(input, np.ndarray)
        if self.type_in != "np":
            raise ValueError(f"{self.__class__.__name__}'s input must be np.ndarray")
        assert len(input.shape) == 2
        assert input.dtype in [float, np.float16, np.float32, np.float64]
        fit_min = np.nanmin(input, axis=0) - float(self.add_value)
        fit_max = np.nanmax(input, axis=0) + float(self.add_value)
        self.fit_values = np.stack([fit_min, fit_max]).astype(input.dtype)
    def call_main(self, input: np.ndarray):
        assert self.fit_values is not None
        output = input # Don't use copy()
        boolwk = np.isnan(output)
        mask_0, mask_1 = np.where(boolwk)
        fill = self.fit_values[np.random.randint(0, 2, mask_1.shape[0]), mask_1]
        output[boolwk] = fill
        return output

class ProcReplaceValue(BaseProc):
    def __init__(self, replace_value: dict, columns: int | str | list[int | str]=None, **kwargs):
        assert isinstance(replace_value, dict) and len(replace_value) > 0
        super().__init__(**kwargs)
        # dict has 2 types
        self.is_dict_rep = False
        if isinstance(list(replace_value.values())[0], dict):
            assert columns is None # No need to be set.
            for x, y in replace_value.items():
                assert check_type(x, [int, str]) # x means col name or col index
                assert check_type(y, dict)
                for a, b in y.items():
                    assert check_type(a, [int, str, float])
                    assert check_type(b, [int, str, float])
            self.is_dict_rep = True
        else:
            for x, y in replace_value.items():
                assert check_type(x, [int, str, float])
                assert check_type(y, [int, str, float])
        if columns is not None:
            if check_type(columns, [int, str]): columns = [columns, ]
            assert check_type_list(columns, [int, str])
        self.replace_value = replace_value
        self.columns       = columns
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in == "pd":
            if self.is_dict_rep:
                assert input.columns.isin(list(self.replace_value.keys())).sum() == len(self.replace_value)
            else:
                if self.columns is not None:
                    assert input.columns.isin(self.columns).sum() == len(self.columns)
        elif self.type_in == "pl":
            if self.is_dict_rep:
                assert np.isin(input.columns, list(self.replace_value.keys())).sum() == len(self.replace_value)
            else:
                if self.columns is not None:
                    assert np.isin(input.columns, self.columns).sum() == len(self.columns)
        else:
            assert len(input.shape) == 2
            if self.is_dict_rep:
                for x, _ in self.replace_value.items():
                    assert isinstance(x, int) and x < input.shape[-1]
            else:
                if self.columns is not None:
                    for x in self.columns:
                        assert isinstance(x, int) and x < input.shape[-1]
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        output = input
        if self.type_in == "pd":
            if self.is_dict_rep:
                dfwk   = pd.concat([output[x].replace(dictwk).copy() for x, dictwk in self.replace_value.items()], axis=1, ignore_index=False, sort=False)
                output = pd.concat([output.loc[:, ~output.columns.isin(dfwk.columns)], dfwk], axis=1, ignore_index=False, sort=False)
            else:
                if self.columns is None:
                    output = output.replace(self.replace_value)
                else:
                    dfwk   = output.loc[:, self.columns].replace(self.replace_value).copy()
                    output = pd.concat([output.loc[:, ~output.columns.isin(dfwk.columns)], dfwk], axis=1, ignore_index=False, sort=False)
        elif self.type_in == "pl":
            if self.is_dict_rep:
                output = output.with_columns([pl.col(x).replace(dictwk) for x, dictwk in self.replace_value.items()])
            else:
                if self.columns is None:
                    output = output.with_columns(pl.all().replace(self.replace_value))
                else:
                    output = output.with_columns([pl.col(x).replace(self.replace_value) for x in self.columns])
        else:
            if self.is_dict_rep:
                for i_col, dictwk in self.replace_value.items():
                    ndf = output[:, i_col]
                    for x, y in dictwk.items():
                        ndf[ndf == x] = y
            else:
                if self.columns is None:
                    for x, y in self.replace_value.items():
                        output[output == x] = y
                else:
                    ndf = output[:, self.columns]
                    for x, y in self.replace_value.items():
                        ndf[ndf == x] = y
        return output

class ProcReplaceInf(ProcReplaceValue):
    def __init__(self, posinf: float=float("nan"), neginf: float=float("nan"), **kwargs):
        assert check_type(posinf, float)
        assert check_type(neginf, float)
        super().__init__(replace_value={float("inf"): posinf, float("-inf"): neginf}, **kwargs)
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in == "np":
            output = np.nan_to_num(input, nan=float("nan"), posinf=self.replace_value[float("inf")], neginf=self.replace_value[float("-inf")])
        else:
            output = super().call_main(input)
        return output

class ProcToValues(BaseProc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def fit_main(self, input: pd.DataFrame | pl.DataFrame):
        assert self.type_in in ["pd", "pl"]
        assert check_type(input, [pd.DataFrame, pl.DataFrame])
    def call_main(self, input):
        if self.type_in == "pd":
            output = input.values
        else:
            output = input.to_numpy()
        return output

class ProcMap(BaseProc):
    def __init__(self, values: dict, column: int | str, fill_null: int | float | str=float("nan"), **kwargs):
        assert isinstance(values, dict) and len(values) > 0
        assert check_type(column, [int, str])
        assert check_type(fill_null, [int, float, str])
        super().__init__(**kwargs)
        self.column    = column
        self.values    = values
        self.fill_null = fill_null
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in in ["pd", "pl"]:
            assert self.column in input.columns
            assert not "__work" in input.columns
        else:
            assert len(input.shape) == 2
            assert isinstance(self.column, int)
    def call_main(self, input):
        output = input
        if   self.type_in == "pd":
            output[self.column] = output[self.column].map(self.values).fillna(self.fill_null)
        elif self.type_in == "pl":
            columns = output.columns.copy()
            output  = output.join(pl.DataFrame([[x, y] for x, y in self.values.items()], strict=False, orient="row", schema=[self.column, "__work"]), how="left", on=self.column)
            output  = output.rename({self.column: "__tmp", "__work": self.column}).select(columns)
            output  = output.with_columns(pl.col(self.column).fill_nan(None).fill_null(self.fill_null))
        else:
            ndf = output[:, self.column].copy()
            ndf = np.vectorize(lambda x: self.values.get(x))(ndf)
            ndf[ndf == None] = self.fill_null
            output[:, self.column] = ndf
        return output

class ProcAsType(BaseProc):
    def __init__(self, to_type: type, columns: str | list[str]=None, **kwargs):
        assert isinstance(to_type, type)
        if isinstance(columns, str): columns = [columns, ]
        assert columns is None or check_type_list(columns, str)
        super().__init__(**kwargs)
        self.to_type = to_type
        self.columns = columns
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if   self.type_in == "pd":
            for x in self.columns: assert x in input.columns
            assert self.to_type in [int, float, str, np.int32, np.int64, np.float16, np.float32, np.float64]
        elif self.type_in == "pl":
            for x in self.columns: assert x in input.columns
            assert self.to_type in [int, float, str, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.String]
        else:
            assert self.columns is None # np.ndarray has only 1 type.
            assert self.to_type in [int, float, str, np.int32, np.int64, np.float16, np.float32, np.float64]
    def call_main(self, input):
        output = input
        if   self.type_in == "pd":
            dfwk   = output.loc[:, self.columns].astype(self.to_type)
            output = pd.concat([output.loc[:, ~output.columns.isin(dfwk.columns)], dfwk], axis=1, ignore_index=False, sort=False)
        elif self.type_in == "pl":
            output = output.with_columns([pl.col(x).cast(self.to_type) for x in self.columns])
        else:
            output = output.astype(self.to_type)
        return output

class ProcReshape(BaseProc):
    def __init__(self, reshape: tuple, **kwargs):
        assert isinstance(reshape, tuple)
        super().__init__(**kwargs)
        self.reshape = reshape
    def fit_main(self, input: np.ndarray):
        if self.type_in != "np":
            raise TypeError(f"{self.__class__.__name__}'s input must be np.ndarray")
    def call_main(self, input: np.ndarray):
        return input.reshape(*self.reshape)

class ProcDropNa(BaseProc):
    def __init__(self, columns: str | list[str], **kwargs):
        if isinstance(columns, str): columns = [columns, ]
        assert check_type_list(columns, str)
        super().__init__(**kwargs)
        self.columns = columns
    def fit_main(self, input: pd.DataFrame | pl.DataFrame):
        if not self.type_in in ["pd", "pl"]:
            raise TypeError(f"{self.__class__.__name__}'s input must be pd.DataFrame or pl.DataFrame")
        for x in self.columns: assert x in input.columns
    def call_main(self, input: pd.DataFrame | pl.DataFrame):
        if self.type_in == "pd":
            output = input.dropna(subset=self.columns)
        else:
            output = output.filter([pl.col(x).fill_nan(None).is_not_null() for x in self.columns])
        return output

class ProcCondition(BaseProc):
    def __init__(self, query_string: str, **kwargs):
        assert isinstance(query_string, str)
        super().__init__(**kwargs)
        self.query_string = query_string
    def fit_main(self, input: pd.DataFrame | pl.DataFrame):
        assert check_type(input, [pd.DataFrame, pl.DataFrame])
        if not self.type_in in ["pd", "pl"]:
            raise TypeError(f"{self.__class__.__name__}'s input must be pd.DataFrame or pl.DataFrame")
    def call_main(self, input: pd.DataFrame | pl.DataFrame):
        if self.type_in == "pd":
            ndf_bool = query(input, self.query_string)
            output   = input.loc[ndf_bool, :]
        else:
            output = pl.SQLContext(test=input, eager=True).execute(f"select * from test where {self.query_string}")
        return output

class ProcDigitize(BaseProc):
    """
    >>> np.digitize([-2,1,2,3,4], [-1,1,2])
    array([0, 2, 3, 3, 3])
    """
    def __init__(self, bins: list[int | float], is_percentile: bool=False, **kwargs):
        assert check_type_list(bins, [int, float])
        assert isinstance(is_percentile, bool)
        super().__init__(**kwargs)
        self.bins  = bins
        self._bins = []
        self.is_percentile = is_percentile
    def fit_main(self, input: np.ndarray):
        assert isinstance(input, np.ndarray)
        assert self.type_in == "np"
        if self.is_percentile:
            self._bins = [np.percentile(input, x) for x in self.bins]
        else:
            self._bins = self.bins
    def call_main(self, input: np.ndarray):
        return np.digitize(input, self._bins)

class ProcEval(BaseProc):
    def __init__(self, eval_string: str, **kwargs):
        assert isinstance(eval_string, str)
        super().__init__(**kwargs)
        self.eval_string = eval_string
    def fit_main(self, *args, **kwargs):
        return None
    def call_main(self, input):
        return eval(self.eval_string, {"pd": pd, "np": np, "__input": input}, {})
