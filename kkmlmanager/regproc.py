import numpy as np
import pandas as pd
import polars as pl

# local package
import kkmlmanager.procs as kkprocs
from kkmlmanager.procs import BaseProc, info_columns
from kkmlmanager.util.com import check_type, check_type_list
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
        self.initialize()
    
    def __str__(self):
        return (
            f"{__class__.__name__}(n_jobs={self.n_jobs}, is_auto_colslct={self.is_auto_colslct}, is_fit={self.is_fit}, " + 
            f"shape={self.shape}), procs={[str(x) for x in self.procs]}"
        )

    def initialize(self):
        self.is_fit   = False
        self.is_check = True
        self.shape: pd.Index | tuple = None
        self.check_proc(self.is_check)
    
    def register(self, proc: str | BaseProc, *args, **kwargs):
        if isinstance(proc, str):
            if "n_jobs" not in kwargs: kwargs["n_jobs"] = self.n_jobs
            proc = getattr(kkprocs, proc)(*args, **kwargs)
        assert isinstance(proc, BaseProc)
        self.procs.append(proc)
        self.initialize()

    def check_proc(self, is_check: bool):
        assert isinstance(is_check, bool)
        self.is_check = is_check
        for proc in self.procs:
            proc.is_check = is_check
    
    def fit(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, check_inout: list[str]=None):
        LOGGER.info("START")
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        if check_inout is None: check_inout = []
        assert check_type_list(check_inout, str)
        for x in check_inout: assert x in ["class", "row", "col"]
        self.shape = info_columns(input)
        output     = input.copy()
        LOGGER.info(f"input: {__class__.info_value(output)}")
        for proc in self.procs:
            LOGGER.info(f"proc: {proc} fit")
            output = proc.fit(output)
            LOGGER.info(f"output: {__class__.info_value(output)}")
        for x in check_inout:
            if   x == "class": type(input) == type(output)
            elif x == "row": input.shape[0]  == output.shape[0]
            elif x == "col": input.shape[-1] == output.shape[-1]
        self.is_fit = True
        LOGGER.info("END")
        return output
    
    def __call__(self, input: pd.DataFrame | np.ndarray | pl.DataFrame, exec_procs: list[int]=[]):
        LOGGER.info("START")
        assert self.is_fit
        assert check_type(input, [pd.DataFrame, np.ndarray, pl.DataFrame])
        assert check_type_list(exec_procs, int)
        if isinstance(input, pd.DataFrame):
            if self.is_auto_colslct:
                input = input.loc[:, self.shape]
            if self.is_check:
                assert input.columns.isin(self.shape).sum() == self.shape.shape[0]
        elif isinstance(input, pl.DataFrame):
            if self.is_auto_colslct:
                input = input.select(self.shape)
            if self.is_check:
                assert input.columns == self.shape
        else:
            if self.is_check:
                assert input.shape[1:] == self.shape
        output = input.copy()
        LOGGER.info(f"input: {__class__.info_value(output)}")
        if len(exec_procs) == 0: procs = self.procs
        else: procs = [self.procs[i] for i in exec_procs]
        for proc in procs:
            LOGGER.info(f"proc: {proc}")
            output = proc(output, n_jobs=self.n_jobs)
            LOGGER.info(f"output: {__class__.info_value(output)}")
        LOGGER.info("END")
        return output

    @classmethod
    def info_value(cls, value: pd.DataFrame | np.ndarray | pl.DataFrame):
        if isinstance(value, pd.DataFrame):
            return f"pd: {value.shape}"
        elif isinstance(value, pl.DataFrame):
            return f"pl: {value.shape}"
        elif isinstance(value, np.ndarray):
            return f"nd: {value.shape}, dtype: {value.dtype}"
        else:
            raise Exception(f"value: {type(value)} is not expected type.")
