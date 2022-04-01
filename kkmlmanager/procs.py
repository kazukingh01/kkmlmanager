import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.decomposition import PCA

# local package
from kkmlmanager.util.dataframe import astype_faster, query
from kkmlmanager.util.com import check_type, check_type_list
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "BaseProc",
    "ProcMinMaxScaler",
    "ProcStandardScaler",
    "ProcRankGauss",
    "ProcPCA",
    "ProcOneHotEncoder",
    "ProcFillNa",
    "ProcFillNaMinMax",
    "ProcReplaceValue",
    "ProcReplaceInf",
    "ProcToValues",
    "ProcDictMap",
    "ProcAsType",
    "ProcReshape",
    "ProcDropNa",
    "ProcCondition",
]


class BaseProc:
    def __init__(self, n_jobs: int=1):
        assert isinstance(n_jobs, int) and n_jobs >= 1 
        self.is_check = False
        self.is_fit   = False
        self.is_df    = False
        self.n_jobs   = n_jobs    
    def fit(self, input: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
        assert check_type(input, [pd.DataFrame, np.ndarray])
        self.is_df = isinstance(input, pd.DataFrame)
        if self.is_df: self.shape_in = input.columns.copy()
        else:          self.shape_in = input.shape[1:]
        output = self.fit_main(input, *args, **kwargs)
        self.is_fit = True
        if output is None:
            output = self(input, *args, **kwargs)
        is_df = isinstance(output, pd.DataFrame)
        if is_df: self.shape_out = output.columns.copy()
        else:     self.shape_out = output.shape[1:]
        return output
    def __call__(self, input: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
        if not self.is_fit: raise Exception("You must use 'fit' first.")
        if self.is_check:
            if self.is_df: assert self.shape_in == input.columns
            else:          assert self.shape_in == input.shape[1:]
        output = self.call_main(input, *args, **kwargs)
        if self.is_check:
            if self.is_df: assert self.shape_out == output.columns
            else:          assert self.shape_out == output.shape[1:]
        return output
    def __str__(self):
        raise NotImplementedError()
    def fit_main(self):
        raise NotImplementedError()
    def call_main(self):
        raise NotImplementedError()

class ProcSKLearn(BaseProc):
    def __init__(self, class_proc, *args, n_jobs: int=1, **kwargs):
        assert isinstance(class_proc, type)
        super().__init__(n_jobs=n_jobs)
        self.proc = class_proc(*args, **kwargs)
    def __str__(self): return str(self.proc)
    def fit_main(self, input):
        return self.proc.fit_transform(input)
    def call_main(self, input):
        return self.proc.transform(input)

class ProcMinMaxScaler(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(MinMaxScaler, *args, **kwargs)

class ProcStandardScaler(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(StandardScaler, *args, **kwargs)

class ProcRankGauss(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(QuantileTransformer, *args, **kwargs)

class ProcPCA(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(PCA, *args, **kwargs)

class ProcOneHotEncoder(ProcSKLearn):
    def __init__(self, *args, **kwargs):
        super().__init__(OneHotEncoder, *args, **kwargs)
    def fit_main(self, input):
        return self.proc.fit_transform(input).toarray()
    def call_main(self, input):
        return self.proc.transform(input).toarray()

class ProcFillNa(BaseProc):
    def __init__(self, fill_value: Union[str, int, float], **kwargs):
        assert check_type(fill_value, [str, int, float, list, np.ndarray])
        if isinstance(fill_value, str):
            assert fill_value in ["mean", "max", "min", "median"]
        super().__init__(**kwargs)
        self.fill_value = fill_value
        self.fit_values = None
    def __str__(self):
        return f'{self.__class__.__name__}(fill_value: {self.fill_value})'
    def fit_main(self, input):
        if self.is_df:
            if isinstance(self.fill_value, str):
                self.fit_values = getattr(input, self.fill_value)(axis=0).values
        else:
            assert len(input.shape) == 2
            if isinstance(self.fill_value, str):
                self.fit_values = getattr(np, f"nan{self.fill_value}")(input, axis=0)
        if isinstance(self.fill_value, list):
            assert check_type_list(self.fill_value, [int, float])
            self.fit_values = np.array(self.fill_value)
            assert self.fit_values.shape[0] == input.shape[-1]
        elif isinstance(self.fill_value, np.ndarray):
            self.fit_values = self.fill_value
            assert self.fit_values.shape[0] == input.shape[-1]
        else:
            self.fit_values = self.fill_value
        assert check_type(self.fit_values, [int, float, np.ndarray])
    def call_main(self, input):
        output = input # Don't use copy()
        if self.is_df:
            if isinstance(self.fit_values, np.ndarray):
                output = output.fillna({x: y for x, y in zip(output.columns, self.fit_values)})
            else:
                output = output.fillna(self.fit_values)
        else:
            if isinstance(self.fit_values, np.ndarray):
                mask = np.isnan(input)
                fill = np.tile(self.fit_values, (input.shape[0], 1))
                output[mask] = fill[mask]
            else:
                output = np.nan_to_num(input, nan=self.fit_values)
            pass
        return output

class ProcFillNaMinMax(BaseProc):
    def __init__(self, add_value: float=1.0, **kwargs):
        assert isinstance(add_value, float)
        super().__init__(**kwargs)
        self.add_value  = add_value
        self.fit_values = np.zeros(0)
    def __str__(self):
        return f'{self.__class__.__name__}()'
    def fit_main(self, input):
        if self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be np.ndarray")
        assert len(input.shape) == 2
        fit_min = np.nanmin(input, axis=0) - self.add_value
        fit_max = np.nanmax(input, axis=0) + self.add_value
        self.fit_values = np.stack([fit_min, fit_max])
    def call_main(self, input):
        output = input # Don't use copy()
        mask_0, mask_1 = np.where(np.isnan(output))
        fill = self.fit_values[np.random.randint(0, 2, mask_1.shape[0]), mask_1]
        output[mask_0, mask_1] = fill
        return output

class ProcReplaceValue(BaseProc):
    def __init__(
        self, target_value: Union[int, float, str], replace_value: Union[int, float, str], 
        indexes: Union[int, str, List[int], List[str]]=None, **kwargs
    ):
        assert check_type(target_value,  [int, float, str])
        assert check_type(replace_value, [int, float, str])
        if indexes is not None:
            if check_type(indexes, [int, str]): indexes = [indexes, ]
            assert check_type_list(indexes, [int, str])
        super().__init__(**kwargs)
        self.target_value  = target_value
        self.replace_value = replace_value
        self.indexes       = indexes
    def __str__(self):
        return f'{self.__class__.__name__}(target_value: {self.target_value}, replace_value: {self.replace_value})'
    def fit_main(self, input):
        if not self.is_df:
            assert len(input.shape) == 2
        if self.indexes is None:
            self.indexes = slice(None)
        else:
            if self.is_df:
                assert check_type_list(self.indexes, str)
            else:
                assert check_type_list(self.indexes, int)
    def call_main(self, input):
        output = input
        if self.is_df:
            output.loc[:, self.indexes] = output.loc[:, self.indexes].replace(self.target_value, self.replace_value)
        else:
            ndf = output[:, self.indexes]
            ndf[ndf == self.target_value] = self.replace_value
            output[:, self.indexes] = ndf
        return output

class ProcReplaceInf(BaseProc):
    def __init__(self, posinf: float=float("nan"), neginf: float=float("nan"), **kwargs):
        assert check_type(posinf, float)
        assert check_type(neginf, float)
        super().__init__(**kwargs)
        self.posinf = posinf
        self.neginf = neginf
    def __str__(self):
        return f'{self.__class__.__name__}(posinf: {self.posinf}, neginf: {self.neginf})'
    def fit_main(self, input):
        if self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be np.ndarray")
    def call_main(self, input):
        output = input
        output = np.nan_to_num(output, nan=float("nan"), posinf=self.posinf, neginf=self.neginf)
        return output

class ProcToValues(BaseProc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __str__(self):
        return f'{self.__class__.__name__}()'
    def fit_main(self, input):
        if not self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be pd.DataFrame")
    def call_main(self, input):
        return input.values

class ProcDictMap(BaseProc):
    def __init__(self, values: dict, index: Union[int, str], **kwargs):
        assert isinstance(values, dict) and len(values) > 0
        assert check_type(index, [int, str])
        super().__init__(**kwargs)
        self.index  = index
        self.values = values
    def __str__(self):
        return f'{self.__class__.__name__}(index: {self.index}, values: {self.values})'
    def fit_main(self, input):
        if self.is_df:
            assert isinstance(self.index, str)
        else:
            assert len(input.shape) == 2
            assert isinstance(self.index, int)
    def call_main(self, input):
        output = input
        if self.is_df:
            output[self.index] = output[self.index].map(self.values)
        else:
            ndf = output[:, self.index].copy()
            ndf = np.vectorize(lambda x: self.values.get(x))(ndf)
            ndf[ndf == None] = float("nan")
            output[:, self.index] = ndf
        return output

class ProcAsType(BaseProc):
    def __init__(self, convert_type: type, indexes: Union[str, List[str]]=None, batch_size: int=1, **kwargs):
        assert isinstance(convert_type, type)
        super().__init__(**kwargs)
        self.convert_type = convert_type
        self.indexes      = indexes
        self.batch_size   = batch_size
    def __str__(self):
        return f'{self.__class__.__name__}(convert_type: {self.convert_type})'
    def fit_main(self, input):
        if self.is_df:
            if self.indexes is None:
                self.indexes = slice(None)
            else:
                if isinstance(self.indexes, str):
                    self.indexes = [self.indexes, ]
                assert check_type_list(self.indexes, str)
        else:
            assert self.indexes is None
    def call_main(self, input):
        output = input
        if self.is_df:
            output = astype_faster(output, list_astype=[{"from": self.indexes, "to": self.convert_type}], batch_size=self.batch_size, n_jobs=self.n_jobs)
        else:
            output = output.astype(self.convert_type)
        return output

class ProcReshape(BaseProc):
    def __init__(self, reshape: tuple, **kwargs):
        assert isinstance(reshape, tuple)
        super().__init__(**kwargs)
        self.reshape = reshape
    def __str__(self):
        return f'{self.__class__.__name__}(reshape: {self.reshape})'
    def fit_main(self, input):
        if self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be np.ndarray")
    def call_main(self, input):
        return input.reshape(*self.reshape)

class ProcDropNa(BaseProc):
    def __init__(self, columns: Union[str, List[str]], **kwargs):
        if isinstance(columns, str): columns = [columns, ]
        assert check_type_list(columns, str)
        super().__init__(**kwargs)
        self.columns = columns
    def __str__(self):
        return f'{self.__class__.__name__}(columns: {self.columns})'
    def fit_main(self, input):
        if not self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be pd.DataFrame")
    def call_main(self, input):
        return input.dropna(subset=self.columns)

class ProcCondition(BaseProc):
    def __init__(self, query_string: str, **kwargs):
        assert isinstance(query_string, str)
        super().__init__(**kwargs)
        self.query_string = query_string
    def __str__(self):
        return f'{self.__class__.__name__}(query_string: {self.query_string})'
    def fit_main(self, input):
        if not self.is_df:
            raise Exception(f"{self.__class__.__name__}'s input must be pd.DataFrame")
    def call_main(self, input):
        ndf_bool = query(input, self.query_string)
        return input.loc[ndf_bool, :]
