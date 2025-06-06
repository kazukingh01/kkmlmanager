import json
import numpy as np
import pandas as pd
import polars as pl

# local package
from . import procs as kkprocs
from .procs import BaseProc, info_columns
from .util.com import check_type, check_type_list
from kklogger import set_logger


__all__ = [
    "RegistryProc",
]


LOGGER = set_logger(__name__)


class RegistryProc(object):
    def __init__(self, n_jobs: int=1, is_auto_colslct: bool=False):
        """
        Usage::
            >>> 
        """
        assert isinstance(n_jobs, int) and n_jobs >= 1
        assert isinstance(is_auto_colslct, bool)
        super().__init__()
        self.procs: list[BaseProc] = []
        self.n_jobs                = n_jobs
        self.is_auto_colslct       = is_auto_colslct
        self.is_fit                = False
        self.shape: list[str | int] | tuple = None
        self.initialize()

    def __str__(self):
        return (
            f"{__class__.__name__}(n_jobs={self.n_jobs}, is_auto_colslct={self.is_auto_colslct}, is_fit={self.is_fit}, " + 
            f"shape={self.shape[:5]}...), procs={[str(x) for x in self.procs]}"
        )

    def __repr__(self):
        return self.__str__()

    def initialize(self):
        self.is_fit = False
        self.shape: list[str | int] | tuple = None
        self.check_proc(True) # Default True.

    def register(self, proc: str | BaseProc, *args, **kwargs):
        if isinstance(proc, str):
            if "n_jobs" not in kwargs: kwargs["n_jobs"] = self.n_jobs
            proc = getattr(kkprocs, proc)(*args, **kwargs)
        assert isinstance(proc, BaseProc)
        self.procs.append(proc)
        self.initialize()

    def check_proc(self, is_check: bool):
        assert isinstance(is_check, bool)
        for proc in self.procs:
            proc.is_check = is_check
    
    def fit(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, check_inout: list[str]=None, is_return_index: bool=False):
        LOGGER.info("START")
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        if check_inout is None: check_inout = []
        assert check_type_list(check_inout, str)
        for x in check_inout: assert x in ["class", "row", "col"]
        assert isinstance(is_return_index, bool)
        self.shape = info_columns(input)
        if isinstance(input, pl.DataFrame):
            output = input.clone()
            if is_return_index:
                output = output.with_columns(pl.Series(np.arange(output.shape[0], dtype=int)).alias("__indexes"))
        else:
            output = input.copy()
        LOGGER.info(f"input: {__class__.info_value(output)}")
        for proc in self.procs:
            LOGGER.info(f"proc: {proc} fit")
            output = proc.fit(output)
            LOGGER.info(f"output: {__class__.info_value(output)}")
        for x in check_inout:
            if   x == "class":
                if type(input) != type(output):
                    raise TypeError(f"Type is different between input and output. input: {type(input)}, output: {type(output)}")
            elif x == "row":
                if input.shape[0] != output.shape[0]:
                    raise TypeError(f"Shape is different between input and output. input: {input.shape}, output: {output.shape}")
            elif x == "col":
                if isinstance(input, (pd.DataFrame, pl.DataFrame)) and isinstance(output, (pd.DataFrame, pl.DataFrame)):
                    if list(input.columns) != list(output.columns):
                        raise TypeError(f"Columns is different between input and output. input: {input.shape}, output: {output.shape}")
                else:
                    if input.shape[-1] != output.shape[-1]:
                        raise TypeError(f"Shape is different between input and output. input: {input.shape}, output: {output.shape}")
        if is_return_index:
            if type(input) != type(output):
                raise TypeError(
                    f"Type is different between input and output. input: {type(input)}, output: {type(output)}. " + 
                    f"if is_return_index = true, input and output class must be same."
                )
            if isinstance(output, pl.DataFrame):
                indexes = output["__indexes"].to_numpy()
                assert indexes.dtype in [int, np.int32, np.int64]
                output = output.select([x for x in output.columns if x != "__indexes"])
            else:
                indexes = output.index.copy()
        self.is_fit = True
        LOGGER.info("END")
        if is_return_index:
            return output, indexes
        else:
            return output
    
    def __call__(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, exec_procs: list[int]=[], is_return_index: bool=False):
        LOGGER.info("START")
        assert self.is_fit
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        assert check_type_list(exec_procs, int)
        if isinstance(input, pd.DataFrame):
            if self.is_auto_colslct:
                input = input.loc[:, self.shape]
        elif isinstance(input, pl.DataFrame):
            if self.is_auto_colslct:
                input = input.select(self.shape)
        else:
            assert self.is_auto_colslct == False
        if isinstance(input, pl.DataFrame):
            output = input.clone()
            if is_return_index:
                output = output.with_columns(pl.Series(np.arange(output.shape[0], dtype=int)).alias("__indexes"))
        else:
            output = input.copy()
        LOGGER.info(f"input: {__class__.info_value(output)}")
        if len(exec_procs) == 0: procs = self.procs
        else: procs = [self.procs[i] for i in exec_procs]
        for proc in procs:
            LOGGER.info(f"proc: {proc}")
            output = proc(output, n_jobs=self.n_jobs)
            LOGGER.info(f"output: {__class__.info_value(output)}")
        if is_return_index:
            if type(input) != type(output):
                raise TypeError(
                    f"Type is different between input and output. input: {type(input)}, output: {type(output)}. " + 
                    f"if is_return_index = true, input and output class must be same."
                )
            if isinstance(output, pl.DataFrame):
                indexes = output["__indexes"].to_numpy()
                assert indexes.dtype in [int, np.int32, np.int64]
                output = output.select([x for x in output.columns if x != "__indexes"])
            else:
                indexes = output.index.copy()
        LOGGER.info("END")
        if is_return_index:
            return output, indexes
        else:
            return output

    @classmethod
    def info_value(cls, value: pd.DataFrame | np.ndarray | pl.DataFrame):
        if isinstance(value, pd.DataFrame):
            return f"pd: {value.shape}"
        elif isinstance(value, pl.DataFrame):
            return f"pl: {value.shape}"
        elif isinstance(value, np.ndarray):
            return f"np: {value.shape}, dtype: {value.dtype}"
        else:
            raise Exception(f"value: {type(value)} is not expected type.")

    def to_dict(self) -> dict:
        return {
            "procs": [x.to_dict() for x in self.procs],
            "n_jobs": self.n_jobs,
            "is_auto_colslct": self.is_auto_colslct,
            "is_fit": self.is_fit,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, dict_regproc: dict):
        assert isinstance(dict_regproc, dict)
        ins        = cls(n_jobs=dict_regproc["n_jobs"], is_auto_colslct=dict_regproc["is_auto_colslct"])
        ins.procs  = [BaseProc.from_dict(x) for x in dict_regproc["procs"]]
        ins.is_fit = dict_regproc["is_fit"]
        ins.shape  = dict_regproc["shape"]
        return ins
    
    def to_json(self, indent: int=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def load_from_json(cls, json_regproc: str):
        assert isinstance(json_regproc, str)
        return cls.from_dict(json.loads(json_regproc))
