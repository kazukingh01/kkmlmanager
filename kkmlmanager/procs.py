import datetime
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
from .util.dataframe import query
from .util.com import check_type, check_type_list


__all__ = [
    "info_columns",
    "mask_values_for_json",
    "unmask_values_for_json",
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
    "ProcMapLabelAuto",
    "ProcAsType",
    "ProcReshape",
    "ProcDropNa",
    "ProcCondition",
    "ProcDigitize",
    "ProcEval",
]


DTYPES_PL_NOT_NAN = [pl.String, pl.Categorical, pl.Datetime, pl.Date, pl.Boolean]


def info_columns(input: pd.DataFrame | np.ndarray | pl.DataFrame) -> pd.Index | list | tuple:
    assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
    if isinstance(input, pd.DataFrame):
        assert input.columns.dtype == object
        shape: list = input.columns.tolist().copy()
    elif isinstance(input, pl.DataFrame):
        shape: list = input.columns.copy()
    else:
        shape: tuple = input.shape[1:]
    return shape

def mask_values_for_json(value: int | float | list | dict, is_key: bool=False):
    if is_key:
        if isinstance(value, int):
            return f"__int__({value})"
        elif isinstance(value, float):
            return f"__float__({value})"
    if isinstance(value, list):
        return [mask_values_for_json(x) for x in value]
    elif isinstance(value, dict):
        return {mask_values_for_json(x, is_key=True): mask_values_for_json(y) for x, y in value.items()}    
    elif isinstance(value, float):
        if np.isnan(value):
            return "__nan__"
        elif value == float("inf"):
            return "__inf__"
        elif value == float("-inf"):
            return "__ninf__"
    elif isinstance(value, (datetime.datetime, datetime.date)):
        return f"__datetime__({value.strftime('%Y-%m-%d %H:%M:%S.%f%z')})"
    elif isinstance(value, type):
        if value == int:
            return "__int__"
        elif value == float:
            return "__float__"
        elif value == str:
            return "__str__"
        elif value == bool:
            return "__bool__"
        elif value == np.int8:
            return "__int8__"
        elif value == np.int16:
            return "__int16__"
        elif value == np.int32:
            return "__int32__"
        elif value == np.int64:
            return "__int64__"
        elif value == np.float16:
            return "__float16__"
        elif value == np.float32:
            return "__float32__"
        elif value == np.float64:
            return "__float64__"
        elif value == pl.Int8:
            return "__Int8__"
        elif value == pl.Int16:
            return "__Int16__"
        elif value == pl.Int32:
            return "__Int32__"
        elif value == pl.Int64:
            return "__Int64__"
        elif value == pl.Float32:
            return "__Float32__"
        elif value == pl.Float64:
            return "__Float64__"
    return value

def unmask_values_for_json(value: int | float | list | dict):
    if isinstance(value, list):
        return [unmask_values_for_json(x) for x in value]
    elif isinstance(value, dict):
        return {unmask_values_for_json(x): unmask_values_for_json(y) for x, y in value.items()}
    elif value == "__nan__":
        return float("nan")
    elif value == "__inf__":
        return float("inf")
    elif value == "__ninf__":
        return float("-inf")
    elif value == "__int__":
        return int
    elif value == "__float__":
        return float
    elif value == "__str__":
        return str
    elif value == "__bool__":
        return bool
    elif value == "__int8__":
        return np.int8
    elif value == "__int16__":
        return np.int16
    elif value == "__int32__":
        return np.int32
    elif value == "__int64__":
        return np.int64
    elif value == "__float16__":
        return np.float16
    elif value == "__float32__":
        return np.float32
    elif value == "__float64__":
        return np.float64
    elif value == "__Int8__":
        return pl.Int8
    elif value == "__Int16__":
        return pl.Int16
    elif value == "__Int32__":
        return pl.Int32
    elif value == "__Int64__":
        return pl.Int64
    elif value == "__Float32__":
        return pl.Float32
    elif value == "__Float64__":
        return pl.Float64
    elif isinstance(value, str):
        if value.startswith("__datetime__("):
            try:
                return datetime.datetime.strptime(value, "__datetime__(%Y-%m-%d %H:%M:%S.%f%z)")
            except ValueError:
                return datetime.datetime.strptime(value, "__datetime__(%Y-%m-%d %H:%M:%S.%f)")
        elif value.startswith("__int__("):
            return int(value.split("__int__(")[1].split(")")[0])
        elif value.startswith("__float__("):
            return float(value.split("__float__(")[1].split(")")[0])
    return value

def set_attributes(ins, dict_proc: dict, exclude: list[str]=[]):
    for x, y in {a: b for a, b in dict_proc.items() if a not in exclude}.items():
        setattr(ins, x, unmask_values_for_json(y))

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
        self.shape_in    = None
        self.shape_out   = None
        self.n_jobs      = n_jobs
        self.is_jobs_fix = is_jobs_fix
    def fit(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, *args, **kwargs):
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        self.type_in  = {pd.DataFrame: "pd", np.ndarray: "np", pl.DataFrame: "pl"}[type(input)]
        self.shape_in = info_columns(input)
        output        = self.fit_main(input, *args, **kwargs) # Basically, it's not retured. but some process run "fit" & "transform" at once because it's efficient
        self.is_fit   = True
        if output is None:
            output = self(input, *args, is_check=False, **kwargs)
        self.type_out  = {pd.DataFrame: "pd", np.ndarray: "np", pl.DataFrame: "pl"}[type(output)]
        self.shape_out = info_columns(output)
        return output
    def __call__(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, *args, n_jobs: int=None, is_check: bool=None, **kwargs):
        if is_check is not None:
            assert isinstance(is_check, bool)
        else:
            is_check = self.is_check
        if not self.is_fit:
            raise NotFittedError("You must 'fit' first.")
        if self.is_jobs_fix == False and n_jobs is not None:
            assert isinstance(n_jobs, int) and n_jobs >= -1
            self.n_jobs = n_jobs
        if is_check:
            if self.type_in == "pd":
                assert isinstance(input, pd.DataFrame)
                assert len(self.shape_in) == len(input.columns)
                assert self.shape_in == input.columns.tolist()
            elif self.type_in == "pl":
                assert isinstance(input, pl.DataFrame)
                assert self.shape_in == input.columns
            else:
                assert isinstance(input, np.ndarray)
                assert tuple(self.shape_in) == input.shape[1:] # After from_dict, shape_in is a list
        output = self.call_main(input, *args, **kwargs)
        if is_check:
            if self.type_out == "pd":
                assert len(self.shape_out) == len(output.columns)
                assert self.shape_out == output.columns.tolist()
            elif self.type_out == "pl":
                assert self.shape_out == output.columns
            else:
                assert tuple(self.shape_out) == output.shape[1:] # After from_dict, shape_out is a list
        return output
    def __str__(self):
        attrs_str = ', '.join(
            f'{k}={v!r}' for k, v in vars(self).items()
            if not k in ["shape_in", "shape_out", "is_check", "is_fit", "type_in", "type_out"]
        )
        return f'{self.__class__.__name__}({attrs_str})'
    def __repr__(self):
        return self.__str__()
    def fit_main(self):
        raise NotImplementedError()
    def call_main(self):
        raise NotImplementedError()
    def to_dict(self) -> dict:
        return {
            "__class__":   self.__class__.__name__,
            "is_check":    self.is_check,
            "is_fit":      self.is_fit,
            "type_in":     self.type_in,
            "type_out":    self.type_out,
            "shape_in":    self.shape_in,
            "shape_out":   self.shape_out,
            "n_jobs":      self.n_jobs,
            "is_jobs_fix": self.is_jobs_fix,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        assert isinstance(dict_proc, dict)
        assert "__class__" in dict_proc
        _cls: BaseProc = globals()[dict_proc["__class__"]]
        assert _cls != __class__
        return _cls.from_dict(dict_proc)

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
    def to_dict(self) -> dict:
        raise NotImplementedError()
    @classmethod
    def from_dict(cls, dict_proc: dict):
        raise NotImplementedError()

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
    def __init__(self, fill_value: str | int | float | list | dict, **kwargs):
        assert check_type(fill_value, [str, int, float, list, dict])
        if isinstance(fill_value, str):
            assert fill_value in ["mean", "max", "min", "median"]
        super().__init__(**kwargs)
        self.fill_value = fill_value
        self.fit_values = None
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if isinstance(self.fill_value, str):
            if self.type_in == "pd":
                columns = [x for x, y in input.dtypes.items() if y in [int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]
                self.fit_values = {x: y for x, y in getattr(input[columns], self.fill_value)(axis=0).items()}
            elif self.type_in == "pl":
                columns = [x for x, y in zip(input.columns, input.dtypes) if y in [pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
                input   = input.select(columns)
                self.fit_values = getattr(input, self.fill_value)().to_dicts()[0]
            else:
                assert len(input.shape) == 2
                assert input.dtype in [int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
                self.fit_values = [x for x in getattr(np, f"nan{self.fill_value}")(input, axis=0)]
        elif isinstance(self.fill_value, list):
            for x in self.fill_value: assert x is not None
            assert len(self.fill_value) == input.shape[-1]
            self.fit_values = self.fill_value.copy()
        elif isinstance(self.fill_value, dict):
            assert self.type_in in ["pd", "pl"]
            for x in self.fill_value.keys():
                assert x in input.columns
            self.fit_values = self.fill_value
        else:
            self.fit_values = self.fill_value
        assert check_type(self.fit_values, [int, float, list, dict])
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        output = input # Don't use copy()
        if self.type_in == "pd":
            if isinstance(self.fit_values, list):
                output = output.fillna({x: y for x, y in zip(output.columns, self.fit_values)})
            elif isinstance(self.fit_values, dict):
                output = output.fillna(self.fit_values)
            else:
                columns = [x for x, y in input.dtypes.items() if not isinstance(y, (pd.CategoricalDtype, pd.Categorical))]
                output  = output.fillna({x: self.fit_values for x in columns})
        elif self.type_in == "pl":
            dict_bool = {x: y in DTYPES_PL_NOT_NAN for x, y in zip(output.columns, output.dtypes)}
            if isinstance(self.fit_values, list):
                output = output.with_columns([
                    pl.col(x).fill_null(y) if dict_bool[x] else pl.col(x).fill_nan(None).fill_null(y)
                    for x, y in zip(output.columns, self.fit_values) if y is not None
                ])
            elif isinstance(self.fit_values, dict):
                output = output.with_columns([
                    pl.col(x).fill_null(y) if dict_bool[x] else pl.col(x).fill_nan(None).fill_null(y)
                    for x, y in self.fit_values.items() if y is not None
                ])
            else:
                output = output.fill_nan(None).fill_null(self.fit_values)
        else:
            if isinstance(self.fit_values, list):
                mask = np.isnan(output).copy()
                for i, x in enumerate(mask.T):
                    output[x, i] = self.fit_values[i]
            elif isinstance(self.fit_values, dict):
                raise TypeError(f"type_in: {self.type_in}, fit_values: {self.fit_values} is not supported")
            else:
                output = np.nan_to_num(input, nan=self.fit_values)
            pass
        return output
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "fill_value": mask_values_for_json(self.fill_value),
            "fit_values": mask_values_for_json(self.fit_values),
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["fill_value"]))
        set_attributes(ins, dict_proc, ["__class__", "fill_value"])
        return ins

class ProcFillNaMinMaxRandomly(BaseProc):
    def __init__(self, add_value: int | float=1.0, **kwargs):
        assert check_type(add_value, [int, float])
        super().__init__(**kwargs)
        self.add_value  = add_value
        self.fit_values = None
    def fit_main(self, input: np.ndarray):
        if self.type_in != "np":
            raise TypeError(f"{self.__class__.__name__}'s input must be np.ndarray")
        assert isinstance(input, np.ndarray)
        assert len(input.shape) == 2
        assert input.dtype in [float, np.float16, np.float32, np.float64]
        fit_min = np.nanmin(input, axis=0) - float(self.add_value)
        fit_max = np.nanmax(input, axis=0) + float(self.add_value)
        self.fit_values = [[float(y) for y in x] for x in np.stack([fit_min, fit_max]).astype(input.dtype)]
    def call_main(self, input: np.ndarray):
        assert self.fit_values is not None
        output = input # Don't use copy()
        boolwk = np.isnan(output)
        mask_0, mask_1 = np.where(boolwk)
        fit_values = np.array(self.fit_values, dtype=float)
        fill       = fit_values[np.random.randint(0, 2, mask_1.shape[0]), mask_1]
        output[boolwk] = fill
        return output
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "add_value": mask_values_for_json(self.add_value),
            "fit_values": mask_values_for_json(self.fit_values),
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["add_value"]))
        set_attributes(ins, dict_proc, ["__class__", "add_value"])
        return ins

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
            list_x = list(replace_value.keys())
            list_y = list(replace_value.values())
            assert (
                (check_type_list(list_x, [int, float]) and check_type_list(list_y, [int, float, type(None)])) or
                (check_type_list(list_x, str)          and check_type_list(list_y, str, type(None)))
            )
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
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "replace_value": mask_values_for_json(self.replace_value),
            "columns": self.columns,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["replace_value"]), columns=dict_proc["columns"])
        set_attributes(ins, dict_proc, ["__class__", "replace_value", "columns"])
        return ins

class ProcReplaceInf(ProcReplaceValue):
    def __init__(self, posinf: float=float("nan"), neginf: float=float("nan"), **kwargs):
        assert isinstance(posinf, (float, type(None)))
        assert isinstance(neginf, (float, type(None)))
        super().__init__(replace_value={float("inf"): posinf, float("-inf"): neginf}, **kwargs)
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        super().fit_main(input)
        if self.type_in == "pl":
            self.replace_value = {x: None if isinstance(y, float) and np.isnan(y) else y for x, y in self.replace_value.items()}
        if self.columns is None:
            if self.type_in == "pd":
                self.columns = [x for x, y in input.dtypes.items() if y in [int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]]
            elif self.type_in == "pl":
                self.columns = [x for x, y in zip(input.columns, input.dtypes) if y in [pl.Float32, pl.Float64]]
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in == "np":
            output = np.nan_to_num(input, nan=float("nan"), posinf=self.replace_value[float("inf")], neginf=self.replace_value[float("-inf")])
        else:
            output = super().call_main(input)
        return output
    @classmethod
    def from_dict(cls, dict_proc: dict):
        dictwk = unmask_values_for_json(dict_proc["replace_value"])
        ins    = cls(posinf=dictwk[float("inf")], neginf=dictwk[float("-inf")])
        set_attributes(ins, dict_proc, ["__class__", "replace_value"])
        return ins

class ProcToValues(BaseProc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def fit_main(self, input: pd.DataFrame | pl.DataFrame):
        if self.type_in not in ["pd", "pl"]:
            raise TypeError(f"{self.__class__.__name__}'s input must be pd.DataFrame or pl.DataFrame")
        assert check_type(input, [pd.DataFrame, pl.DataFrame])
    def call_main(self, input: pd.DataFrame | pl.DataFrame):
        output = input.to_numpy()
        return output
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls()
        set_attributes(ins, dict_proc, ["__class__"])
        return ins

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
            dfwk    = pl.DataFrame([[x, y] for x, y in self.values.items()], strict=False, orient="row", schema=[self.column, "__work"])
            dfwk    = dfwk.with_columns(pl.col(self.column).cast(output[self.column].dtype))
            output  = output.join(dfwk, how="left", on=self.column)
            output  = output.rename({self.column: "__tmp", "__work": self.column}).select(columns)
            if output[self.column].dtype in DTYPES_PL_NOT_NAN:
                output = output.with_columns(pl.col(self.column).fill_null(self.fill_null))
            else:
                output = output.with_columns(pl.col(self.column).fill_nan(None).fill_null(self.fill_null))
        else:
            ndf = output[:, self.column].copy()
            ndf = np.vectorize(lambda x: self.values.get(x))(ndf)
            ndf[ndf == None] = self.fill_null
            output[:, self.column] = ndf
        return output
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "values": mask_values_for_json(self.values),
            "column": self.column,
            "fill_null": mask_values_for_json(self.fill_null),
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["values"]), dict_proc["column"], unmask_values_for_json(dict_proc["fill_null"]))
        set_attributes(ins, dict_proc, ["__class__", "values", "column", "fill_null"])
        return ins

class ProcMapLabelAuto(ProcMap):
    def __init__(self, column: int | str, fill_null: int | float | str=float("nan"), **kwargs):
        assert check_type(column, [int, str])
        assert check_type(fill_null, [int, float, str])
        super().__init__({"tmp": 0}, column, fill_null=fill_null, **kwargs)
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        super().fit_main(input) # for checking
        if self.type_in in ["pd", "pl"]:
            values = np.unique(input[self.column].to_numpy())
        else:
            values = np.unique(input[:, self.column])
        assert np.isnan(values).sum() == 0
        assert values.dtype in [int, str, object, np.int16, np.int32, np.int64]
        if values.dtype in [int, np.int16, np.int32, np.int64]:
            self.values = {int(x): i for i, x in enumerate(np.sort(values))}
        else:
            self.values = {x: i for i, x in enumerate(np.sort(values))}
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(dict_proc["column"], unmask_values_for_json(dict_proc["fill_null"]))
        set_attributes(ins, dict_proc, ["__class__", "column", "fill_null"])
        return ins
    
class ProcAsType(BaseProc):
    def __init__(self, to_type: type, columns: str | list[str]=None, **kwargs):
        assert isinstance(to_type, type)
        if isinstance(columns, (int, str)): columns = [columns, ]
        assert columns is None or check_type_list(columns, [int, str])
        super().__init__(**kwargs)
        self.to_type = to_type
        self.columns = columns
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if   self.type_in == "pd":
            if self.columns is None:
                self.columns = input.columns.tolist()
            for x in self.columns: assert x in input.columns
            assert self.to_type in [int, float, str, np.int32, np.int64, np.float16, np.float32, np.float64]
        elif self.type_in == "pl":
            if self.columns is None:
                self.columns = input.columns.copy()
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
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "to_type": mask_values_for_json(self.to_type),
            "columns": self.columns,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["to_type"]), columns=dict_proc["columns"])
        set_attributes(ins, dict_proc, ["__class__", "to_type", "columns"])
        return ins

class ProcReshape(BaseProc):
    def __init__(self, *reshape: tuple, **kwargs):
        if len(reshape) == 1 and isinstance(reshape[0], tuple):
            reshape = reshape[0]
        assert isinstance(reshape, tuple)
        assert check_type_list(reshape, int)
        super().__init__(**kwargs)
        self.reshape = reshape
    def fit_main(self, input: np.ndarray):
        if self.type_in != "np":
            raise TypeError(f"{self.__class__.__name__}'s input must be np.ndarray")
    def call_main(self, input: np.ndarray):
        return input.reshape(*self.reshape)
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "reshape": self.reshape,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(tuple(dict_proc["reshape"]) if isinstance(dict_proc["reshape"], list) else dict_proc["reshape"])
        set_attributes(ins, dict_proc, ["__class__", "reshape"])
        return ins

class ProcDropNa(BaseProc):
    def __init__(self, columns: int | str | list[str | int], **kwargs):
        if not isinstance(columns, list): columns = [columns, ]
        assert check_type_list(columns, [int, str])
        super().__init__(**kwargs)
        self.columns = columns
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in in ["pd", "pl"]:
            for x in self.columns: assert x in input.columns
        else:
            assert check_type_list(self.columns, int)
            assert len(input.shape) == 2
            for x in self.columns: assert x < input.shape[-1]
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        if self.type_in == "pd":
            output = input.dropna(subset=self.columns)
        elif self.type_in == "pl":
            dict_bool = {x: y in DTYPES_PL_NOT_NAN for x, y in zip(self.columns, input.select(self.columns).dtypes)}
            output = input.filter([
                pl.col(x).is_not_null() if dict_bool[x] else pl.col(x).fill_nan(None).is_not_null()
                for x in self.columns
            ])
        else:
            ndf_bool = np.isnan(input[:, self.columns].astype(float)).max(axis=-1)
            output   = input[~ndf_bool]
        return output
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "columns": self.columns,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(dict_proc["columns"])
        set_attributes(ins, dict_proc, ["__class__", "columns"])
        return ins

class ProcCondition(BaseProc):
    def __init__(self, query_string: str, **kwargs):
        assert isinstance(query_string, str)
        super().__init__(**kwargs)
        self.query_string = query_string
    def fit_main(self, input: pd.DataFrame | pl.DataFrame):
        if not self.type_in in ["pd", "pl"]:
            raise TypeError(f"{self.__class__.__name__}'s input must be pd.DataFrame or pl.DataFrame")
        assert check_type(input, [pd.DataFrame, pl.DataFrame])
    def call_main(self, input: pd.DataFrame | pl.DataFrame):
        if self.type_in == "pd":
            ndf_bool = query(input, self.query_string)
            output   = input.loc[ndf_bool, :]
        else:
            output = pl.SQLContext(test=input, eager=True).execute(f"select * from test where {self.query_string}")
        return output
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "query_string": self.query_string,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(dict_proc["query_string"])
        set_attributes(ins, dict_proc, ["__class__", "query_string"])
        return ins

class ProcDigitize(BaseProc):
    """
    Not supported
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
        if self.type_in != "np":
            raise TypeError(f"{self.__class__.__name__}'s input must be np.ndarray")
        assert isinstance(input, np.ndarray)
        assert input.dtype in [int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        if self.is_percentile:
            self._bins = [float(np.percentile(input, x)) for x in self.bins]
        else:
            self._bins = self.bins
    def call_main(self, input: np.ndarray):
        return np.digitize(input, self._bins)
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "bins":  self.bins,
            "_bins": self._bins,
            "is_percentile": self.is_percentile,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(unmask_values_for_json(dict_proc["bins"]), dict_proc["is_percentile"])
        set_attributes(ins, dict_proc, ["__class__", "bins", "is_percentile"])
        return ins

class ProcEval(BaseProc):
    def __init__(self, eval_string: str, **kwargs):
        assert isinstance(eval_string, str)
        super().__init__(**kwargs)
        self.eval_string = eval_string
    def fit_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        return None
    def call_main(self, input: pd.DataFrame | np.ndarray | pl.DataFrame):
        return eval(self.eval_string, {"pd": pd, "np": np, "pl": pl, "__input": input}, {})
    def to_dict(self) -> dict:
        return super().to_dict() | {
            "eval_string": self.eval_string,
        }
    @classmethod
    def from_dict(cls, dict_proc: dict):
        ins = cls(dict_proc["eval_string"])
        set_attributes(ins, dict_proc, ["__class__", "eval_string"])
        return ins
