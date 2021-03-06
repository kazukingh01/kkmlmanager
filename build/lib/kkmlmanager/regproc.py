import numpy as np
import pandas as pd
from typing import List, Union

# local package
import kkmlmanager.procs as kkprocs
from kkmlmanager.procs import BaseProc
from kkmlmanager.util.com import check_type, check_type_list
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "RegistryProc",
]


class RegistryProc(object):
    def __init__(self, n_jobs: int=1):
        """
        Usage::
            >>> 
        """
        assert isinstance(n_jobs, int) and n_jobs >= 1
        super().__init__()
        self.procs: List[BaseProc] = []
        self.n_jobs = n_jobs
        self.initialize()
    
    def __str__(self):
        return str([str(x) for x in self.procs])

    def initialize(self):
        self.is_fit   = False
        self.is_check = False
        self.shape    = None
        self.check_proc(self.is_check)
    
    def register(self, proc: Union[str, BaseProc], *args, **kwargs):
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
    
    def fit(self, input: Union[pd.DataFrame, np.ndarray], check_inout: List[str]=None):
        logger.info("START")
        assert check_type(input, [pd.DataFrame, np.ndarray])
        if check_inout is None: check_inout = []
        assert check_type_list(check_inout, str)
        for x in check_inout: assert x in ["class", "row", "col"]
        if isinstance(input, pd.DataFrame):
            assert input.columns.dtype == object
            self.shape = input.columns.copy()
        else:
            self.shape = input.shape[1:]
        output = input.copy()
        logger.info(f"input: {__class__.info_value(output)}")
        for proc in self.procs:
            logger.info(f"proc: {proc} fit")
            output = proc.fit(output)
            logger.info(f"output: {__class__.info_value(output)}")
        for x in check_inout:
            if   x == "class": type(input) == type(output)
            elif x == "row": input.shape[0]  == output.shape[0]
            elif x == "col": input.shape[-1] == output.shape[-1]
        self.is_fit = True
        logger.info("END")
        return output
    
    def __call__(self, input: Union[pd.DataFrame, np.ndarray]):
        logger.info("START")
        assert self.is_fit
        assert check_type(input, [pd.DataFrame, np.ndarray])
        if isinstance(input, pd.DataFrame):
            if self.is_check:
                assert input.columns.isin(self.shape).sum() == self.shape.shape[0]
                output = input.loc[:, self.shape].copy()
            else:
                output = input.copy()
        else:
            if self.is_check:
                assert input.shape[1:] == self.shape
            output = input
        logger.info(f"input: {__class__.info_value(output)}")
        for proc in self.procs:
            logger.info(f"proc: {proc}")
            output = proc(output, n_jobs=self.n_jobs)
            logger.info(f"output: {__class__.info_value(output)}")
        logger.info("END")
        return output

    @classmethod
    def info_value(cls, value: Union[pd.DataFrame, np.ndarray]):
        if isinstance(value, pd.DataFrame):
            return f"df: {value.shape}"
        elif isinstance(value, np.ndarray):
            return f"nd: {value.shape}, dtype: {value.dtype}"
        else:
            raise Exception(f"value: {type(value)} is not expected type.")

