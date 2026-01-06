import sys, pickle, os, copy, datetime, json, re, typing, importlib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from kklogger import set_logger

# local package
from .regproc import RegistryProc
from .features import get_features_by_variance_pl, get_features_by_correlation, get_features_by_randomtree_importance, get_features_by_adversarial_validation
from .eval import eval_model
from .models import MultiModel, Calibrator
from .procs import mask_values_for_json, unmask_values_for_json
from .util.numpy import isin_compare_string
from .util.dataframe import encode_dataframe_to_zip_base64, decode_dataframe_from_zip_base64
from .util.com import check_type_list, correct_dirpath, makedirs, unmask_values, unmask_value_isin_object, PICKLE_PROTOCOL, encode_object, decode_object
LOGGER = set_logger(__name__)


__all__ = [
    "MLManager",
]


DATAFRAME:      type = pd.DataFrame | pl.DataFrame
DATAFRAME_NONE: type = pd.DataFrame | pl.DataFrame | None


class MLManager:
    """
    Usage::
        see: https://github.com/kazukingh01/kkmlmanager/tree/main/tests
    """
    def __init__(
        self,
        # model parameter
        columns_exp: list[str], columns_ans: str | list[str], columns_oth: list[str]=None, is_reg: bool=False, 
        # common parameter
        random_seed: int=1, n_jobs: int=1
    ):
        self.logger = set_logger(f"{__class__.__name__}.{id(self)}", internal_log=True)
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(columns_exp, (list, np.ndarray))
        assert isinstance(columns_ans, (list, np.ndarray, str))
        assert isinstance(columns_oth, (list, np.ndarray, type(None)))
        if isinstance(columns_ans, str): columns_ans = [columns_ans, ]
        columns_oth = [] if columns_oth is None else columns_oth
        assert check_type_list(columns_exp, str)
        assert check_type_list(columns_ans, str)
        assert check_type_list(columns_oth, str)
        assert isinstance(is_reg, bool)
        assert isinstance(random_seed, int) and random_seed >= 0
        assert isinstance(n_jobs, int) and n_jobs >= 1
        self.columns_exp = np.array(columns_exp, dtype=object)
        self.columns_ans = np.array(columns_ans, dtype=object)
        self.columns_oth = np.array(columns_oth, dtype=object)
        self.is_reg      = is_reg
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self.initialize()
        self.logger.info("END", color=["GREEN", "BOLD"])
    
    def __str__(self):
        return f"{__class__.__name__}(model: {self.model}\ncolumns explain: {self.columns_exp}\ncolumns answer: {self.columns_ans}\ncolumns other: {self.columns_oth})"
    
    def __repr__(self):
        return self.__str__()

    def initialize(self) -> typing.Self:
        self.logger.info("START")
        self.model        = None
        self.model_class  = None
        self.model_args   = None
        self.model_kwargs = None
        self.model_post: MultiModel | Calibrator = None
        self.model_func   = None
        self.is_fit       = False
        self.is_postmodel = False
        self.list_cv      = []
        self.list_loop    = []
        self.columns_hist = [self.columns_exp.copy(), ]
        self.columns      = self.columns_exp.copy()
        self.proc_row     = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=False)
        self.proc_exp     = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=True)
        self.proc_ans     = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=True)
        self.proc_check_init()
        self.eval_train_se: pd.Series    = pd.Series(   dtype=object)
        self.eval_train_df: pd.DataFrame = pd.DataFrame(dtype=object)
        self.eval_valid_se: pd.Series    = pd.Series(   dtype=object)
        self.eval_valid_df: pd.DataFrame = pd.DataFrame(dtype=object)
        self.eval_test_se:  pd.Series    = pd.Series(   dtype=object)
        self.eval_test_df:  pd.DataFrame = pd.DataFrame(dtype=object)
        self.logger.info("END")
        return self
    
    def to_dict(self, mode: int=0, savedir: str=None) -> dict:
        self.logger.info("START")
        assert isinstance(mode, int) and mode in [0, 1, 2]
        dictwk = {
            "__BaseModel__": "kkmlmanager.manager.MLManager",
            "model_class": {
                "__name__":   self.model_class.__name__,
                "__binary__": encode_object(self.model_class, mode=0, savedir=None),
            } if self.model_class is not None else None,
            "model_args":    [mask_values_for_json(x) for x in self.model_args] if hasattr(self, "model_args") and self.model_args is not None else None,
            "model_kwargs":  {k: mask_values_for_json(v) for k, v in self.model_kwargs.items()} if hasattr(self, "model_kwargs") and self.model_kwargs is not None else None,
            "model_post":    self.model_post.to_dict(mode=mode, savedir=savedir) if self.model_post is not None else None,
            "model_func":    self.model_func,
            "list_cv":       self.list_cv,
            "list_loop":     self.list_loop,
            "is_fit":        self.is_fit,
            "is_postmodel":  self.is_postmodel,
            "is_reg":        self.is_reg,
            "random_seed":   self.random_seed,
            "n_jobs":        self.n_jobs,
            "columns_hist":  [x.tolist() for x in self.columns_hist],
            "columns":       self.columns.tolist(),
            "columns_exp":   self.columns_exp.tolist(),
            "columns_ans":   self.columns_ans.tolist(),
            "columns_oth":   self.columns_oth.tolist(),
            "proc_row":      self.proc_row.to_dict(),
            "proc_exp":      self.proc_exp.to_dict(),
            "proc_ans":      self.proc_ans.to_dict(),
            "eval_train_se": self.eval_train_se.to_dict(),
            "eval_valid_se": self.eval_valid_se.to_dict(),
            "eval_test_se":  self.eval_test_se. to_dict(),
        }
        for x in dir(self):
            if re.match(r"^eval_valid_se_cv", x) is not None:
                dictwk[x] = getattr(self, x).to_dict()
            elif x in ["eval_adversarial_se"]:
                dictwk[x] = getattr(self, x).to_dict()
        for x in dir(self): # separating to avoid long text goes head
            if (re.match(r"^features_", x) is not None) or (re.match(r"^eval_valid_df_cv", x) is not None):
                if mode != 2:
                    LOGGER.info(f"encode: {x}, it might be slow process...")
                dictwk[x] = encode_object(getattr(self, x), mode=mode, savedir=savedir, func_encode=encode_dataframe_to_zip_base64)
        dictwk["model"] = encode_object(self.model, mode=mode, savedir=savedir) if self.model is not None else None # long text goes last
        for x in dir(self):
            if re.match(r"^model_cv[0-9]+$", x) is not None:
                dictwk[x] = encode_object(getattr(self, x), mode=mode, savedir=savedir)
        self.logger.info("END")
        return dictwk
    
    def to_json(self, indent: int=None, mode: int=0, savedir: str=None):
        self.logger.info("START")
        dictwk = self.to_dict(mode=mode, savedir=savedir)
        self.logger.info("END")
        return json.dumps(dictwk, indent=indent)
    
    @classmethod
    def from_dict(cls, dictwk: dict, n_jobs: int | None=None, basedir: str | None=None):
        LOGGER.info("START")
        assert isinstance(dictwk, dict)
        assert n_jobs  is None or isinstance(n_jobs, int)
        assert basedir is None or isinstance(basedir, str)
        n_jobs = dictwk["n_jobs"] if n_jobs is None else n_jobs
        ins = cls(
            dictwk["columns_exp"], dictwk["columns_ans"], columns_oth=dictwk["columns_oth"],
            is_reg=dictwk["is_reg"], random_seed=dictwk["random_seed"], n_jobs=n_jobs
        )
        exclude_list = ["columns_exp", "columns_ans", "columns_oth", "is_reg", "random_seed", "n_jobs"]
        for x, y in dictwk.items():
            if x in exclude_list: continue
            if x == "model":
                ins.model = decode_object(y, basedir=basedir) if y is not None else None
            elif x == "model_class":
                ins.model_class = decode_object(y["__binary__"]) if y is not None else None
            elif x == "model_args":
                ins.model_args = tuple([unmask_values_for_json(y[i]) for i in range(len(y))]) if y is not None else None
            elif x == "model_kwargs":
                ins.model_kwargs = {k: unmask_values_for_json(v) for k, v in y.items()} if y is not None else None
            elif x == "model_post":
                if y is not None:
                    _path, _cls = y["__BaseModel__"].rsplit(".", 1)
                    _cls = getattr(importlib.import_module(_path), _cls)
                    ins.model_post = _cls.from_dict(y, basedir=basedir)
            elif x == "proc_row":
                ins.proc_row = RegistryProc.from_dict(y)
            elif x == "proc_exp":
                ins.proc_exp = RegistryProc.from_dict(y)
            elif x == "proc_ans":
                ins.proc_ans = RegistryProc.from_dict(y)
            elif x == "eval_train_se":
                ins.eval_train_se = pd.Series(y)
            elif x == "eval_valid_se":
                ins.eval_valid_se = pd.Series(y)
            elif x == "eval_test_se":
                ins.eval_test_se  = pd.Series(y)
            elif x == "eval_adversarial_se":
                ins.eval_adversarial_se = pd.Series(y)
            elif x == "columns":
                ins.columns      = np.array(y, dtype=object)
            elif x == "columns_hist":
                ins.columns_hist = [np.array(tmp, dtype=object) for tmp in y]
            elif (re.match(r"^features_", x) is not None) or (re.match(r"^eval_valid_df_cv", x) is not None):
                try:
                    setattr(ins, x, decode_object(y, basedir=basedir, func_decode=decode_dataframe_from_zip_base64))
                except Exception as e:
                    LOGGER.warning(f"failed to decode {x}: {e}")
                    setattr(ins, x, None)
            elif re.match(r"^eval_valid_se_cv", x) is not None:
                setattr(ins, x, pd.Series(y))
            elif re.match(r"^model_cv[0-9]+$", x) is not None:
                setattr(ins, x, decode_object(y, basedir=basedir))
            else:
                setattr(ins, x, y)
        LOGGER.info("END")
        return ins

    @classmethod
    def from_json(cls, json_str: str, n_jobs: int | None=None, basedir: str | None=None):
        LOGGER.info("START")
        assert isinstance(json_str, str)
        ins = cls.from_dict(json.loads(json_str), n_jobs=n_jobs, basedir=basedir)
        LOGGER.info("END")
        return ins

    def copy(self, is_minimum: bool=False) -> typing.Self:
        assert isinstance(is_minimum, bool)
        if is_minimum:
            ins = self.__class__(
                self.columns_exp.tolist(), self.columns_ans.tolist(), columns_oth=self.columns_oth.tolist(), 
                is_reg=self.is_reg, random_seed=self.random_seed, n_jobs=self.n_jobs
            )
            ins.logger.internal_stream.write(copy.deepcopy(self.logger.internal_stream.getvalue()))
            ins.model_class  = copy.deepcopy(self.model_class)
            ins.model_args   = copy.deepcopy(self.model_args)
            ins.model_kwargs = copy.deepcopy(self.model_kwargs)
            ins.model_func   = copy.deepcopy(self.model_func)
            ins.is_fit       = self.is_fit
            ins.is_postmodel = self.is_postmodel
            ins.columns_hist = copy.deepcopy(self.columns_hist)
            ins.columns      = self.columns.copy()
            ins.proc_row     = copy.deepcopy(self.proc_row)
            ins.proc_exp     = copy.deepcopy(self.proc_exp)
            ins.proc_ans     = copy.deepcopy(self.proc_ans)
            ins.list_cv      = copy.deepcopy(self.list_cv)
            ins.list_loop    = copy.deepcopy(self.list_loop)
            model_mode       = self.get_model_mode()
            if   model_mode == "model_post":
                ins.model      = None
                ins.model_post = copy.deepcopy(self.model_post)
            else:
                if hasattr(self.model, "copy"):
                    ins.model  = self.model.copy()
                else:
                    ins.model  = copy.deepcopy(self.model)
            return ins
        else:
            return copy.deepcopy(self)

    def set_model(self, model, *args, model_func_predict: str | None=None, **kwargs):
        self.logger.info("START")
        self.logger.info(f"model: {model}, args: {args}, model_func_predict: {model_func_predict}, kwargs: {kwargs}")
        assert isinstance(model, type)
        assert isinstance(model_func_predict, (str, type(None)))
        self.model_class  = model
        self.model_args   = args
        self.model_kwargs = kwargs
        if model_func_predict is None:
            if self.is_reg: model_func_predict = "predict"
            else:           model_func_predict = "predict_proba"
        self.model_func   = model_func_predict
        self.reset_model()
        self.logger.info("END")
    
    def reset_model(self):
        self.logger.info("START")
        self.model        = self.model_class(*self.model_args, **self.model_kwargs)
        self.model_post   = None
        self.is_fit       = False
        self.is_postmodel = False
        self.logger.info("END")

    def update_features(self, features: list[str] | np.ndarray):
        self.logger.info("START")
        assert isinstance(features, (np.ndarray, list))
        if isinstance(features, list):
            assert check_type_list(features, str)
            features = np.array(features, dtype=object)
        assert np.all(isin_compare_string(features, self.columns))
        self.columns_hist.append(self.columns.copy())
        self.columns = features.copy()
        self.logger.info(f"columns new: {self.columns.shape}, columns before:{self.columns_hist[-1].shape}")
        self.logger.info("END")

    def cut_features_by_variance(self, df: DATAFRAME_NONE=None, cutoff: float=0.99, ignore_nan: bool=False, n_divide: int=10000):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df, DATAFRAME_NONE)
        assert isinstance(cutoff, float) and 0 < cutoff <= 1.0
        assert isinstance(ignore_nan, bool)
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, ignore_nan: {ignore_nan}")
        attr_name = f"features_var_{ignore_nan}_{str(cutoff)[:5].replace('.', '')}"
        if isinstance(df, pd.DataFrame):
            df = pl.from_dataframe(df)
        if df is not None:
            sebool = get_features_by_variance_pl(df.select(self.columns), cutoff=cutoff, ignore_nan=ignore_nan, n_divide=n_divide)
            setattr(self, attr_name, sebool.copy())
        else:
            try: sebool = getattr(self, attr_name)
            except AttributeError:
                self.logger.warning(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        columns = sebool.index[~sebool].to_numpy(dtype=object)
        assert len(columns) > 0, f"Increase cutoff: {cutoff} to keep at least one feature."
        if df is not None:
            columns_del = self.columns[~isin_compare_string(self.columns, columns)].copy()
            df_del = df.select(columns_del).fill_nan(None)
            for x in columns_del:
                dfwk = df_del[x].value_counts().sort("count")
                self.logger.info(
                    f"feature: {x}, n nan: {df_del[x].null_count()}, max count: {(dfwk[x][-1], dfwk["count"][-1])}, " + 
                    f"unique: {dfwk[x].to_numpy()[-5:]}"
                )
        self.update_features(columns)
        self.logger.info("END", color=["GREEN", "BOLD"])

    def cut_features_by_correlation(
        self, df: DATAFRAME_NONE=None, cutoff: float | None=0.99, sample_size: int | None=None, dtype: str="float16", is_gpu: bool=False, 
        corr_type: str="pearson", batch_size: int=100, min_n: int=10, **kwargs
    ):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df, DATAFRAME_NONE)
        assert cutoff      is None or (isinstance(cutoff,    float) and 0 < cutoff <= 1.0)
        assert sample_size is None or (isinstance(sample_size, int) and 0 < sample_size)
        assert dtype     in ["float16", "float32", "float64"]
        assert corr_type in ["pearson", "spearman", "chatterjee"]
        sample_size = df.shape[0] if sample_size is None and df is not None else sample_size
        if df is not None:
            assert sample_size <= df.shape[0]
            idx = np.random.permutation(np.arange(df.shape[0]))[:sample_size]
            if isinstance(df, pd.DataFrame):
                df = pl.from_dataframe(df)
            df = df[idx][:, self.columns]
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, dtype: {dtype}, is_gpu: {is_gpu}, corr_type: {corr_type}")
        attr_name = f"features_corr_{corr_type}"
        if df is not None:
            df_corr = get_features_by_correlation(
                df, dtype=dtype, is_gpu=is_gpu, corr_type=corr_type, batch_size=batch_size, min_n=min_n, n_jobs=self.n_jobs, **kwargs
            )
            for i in range(df_corr.shape[0]): df_corr.iloc[i:, i] = float("nan")
            setattr(self, attr_name, df_corr.copy().astype(np.float32))
        else:
            try: df_corr = getattr(self, attr_name)
            except AttributeError:
                self.logger.raise_error(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.", exception=AttributeError())
        if cutoff is not None:
            if corr_type == "chatterjee":
                columns_del = df_corr.columns[(df_corr > cutoff).sum(axis=0) > 0].to_numpy(dtype=object)
            else:
                columns_del = df_corr.columns[((df_corr > cutoff) | (df_corr < -cutoff)).sum(axis=0) > 0].to_numpy(dtype=object)
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del:
                sewk  = df_corr.loc[:, x].copy()
                if corr_type == "chatterjee":
                    index = np.where(sewk > cutoff)[0][0]
                else:
                    index = np.where((sewk > cutoff) | (sewk < -cutoff))[0][0]
                self.logger.info(f"feature: {x}, compare: {sewk.index[index]}, corr: {sewk.iloc[index]}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            assert len(columns) > 0, f"Increase cutoff: {cutoff} to keep at least one feature."
            self.update_features(columns)
        else:
            self.logger.info(f"cutoff is not set so no updated columns.")
        self.logger.info("END", color=["GREEN", "BOLD"])

    def cut_features_by_randomtree_importance(
        self, df: DATAFRAME_NONE=None, cutoff: float | None=0.9, max_iter: int=1, min_count: int=100, dtype="float32", **kwargs
    ):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df, DATAFRAME_NONE)
        assert cutoff is None or isinstance(cutoff, float) and 0 < cutoff <= 1.0
        assert isinstance(max_iter,  int) and max_iter  > 0
        assert isinstance(min_count, int) and min_count > 0
        assert isinstance(dtype, str) and dtype in ["float32", "float64"]
        assert len(self.columns_ans.shape) == 1
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, max_iter: {max_iter}, min_count: {min_count}")
        if df is not None:
            if isinstance(df, pd.DataFrame):
                df = pl.from_dataframe(df)
            df_treeimp = get_features_by_randomtree_importance(
                df, self.columns.tolist(), self.columns_ans[0], dtype={"float32": pl.Float32, "float64": pl.Float64}[dtype],
                is_reg=self.is_reg, max_iter=max_iter, min_count=min_count, n_jobs=self.n_jobs, **kwargs
            )
            df_treeimp = df_treeimp.sort_values("ratio", ascending=False)
            self.features_treeimp = df_treeimp.copy()
        else:
            try: df_treeimp = getattr(self, "features_treeimp")
            except AttributeError:
                self.logger.warning(f"features_treeimp is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        columns_sort = df_treeimp.index.to_numpy(dtype=object).copy()
        self.columns = np.concatenate([columns_sort[isin_compare_string(columns_sort, self.columns)], self.columns[~isin_compare_string(self.columns, columns_sort)]])
        if cutoff is not None:
            columns_del = columns_sort[int(len(columns_sort) * cutoff):]
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del: self.logger.info(f"feature: {x}, ratio: {df_treeimp.loc[x, 'ratio']}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            assert len(columns) > 0, f"Increase cutoff: {cutoff} to keep at least one feature."
            self.update_features(columns)
        else:
            self.logger.info(f"cutoff is not set so no updated columns.")
        self.logger.info("END", color=["GREEN", "BOLD"])

    def cut_features_by_adversarial_validation(
        self, df_train: DATAFRAME_NONE=None, df_test: DATAFRAME_NONE=None,
        cutoff: int | float | None=None, thre_count: int | str="mean", n_split: int=5, n_cv: int=5, dtype: str="float32", **kwargs
    ):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_train, DATAFRAME_NONE)
        assert isinstance(df_test,  DATAFRAME_NONE)
        assert type(df_train) == type(df_test)
        assert cutoff is None or isinstance(cutoff, (int, float)) and 0 <= cutoff
        assert thre_count is not None and (isinstance(thre_count, str) or isinstance(thre_count, int))
        if isinstance(thre_count, str): assert thre_count in ["mean"]
        if isinstance(thre_count, int): assert thre_count >= 0
        assert isinstance(n_split, int) and n_split > 0
        assert isinstance(n_cv,    int) and n_cv    > 0
        assert isinstance(dtype, str) and dtype in ["float32", "float64"]
        self.logger.info(
            f"df_train: {df_train.shape if df_train is not None else None}, " + 
            f"df_test: { df_test.shape  if df_test  is not None else None}, " + 
            f"cutoff: {cutoff}, thre_count: {thre_count}, n_split: {n_split}, n_cv: {n_cv}, dtype: {dtype}"
        )
        if df_train is not None:
            if isinstance(df_train, pd.DataFrame):
                df_train = pl.from_dataframe(df_train)
                df_test  = pl.from_dataframe(df_test )
            df_adv, df_pred, se_eval = get_features_by_adversarial_validation(
                df_train, df_test, self.columns.tolist(), columns_ans=None, 
                n_split=n_split, n_cv=n_cv, dtype={"float32": pl.Float32, "float64": pl.Float64}[dtype], n_jobs=self.n_jobs, **kwargs
            )
            df_adv = df_adv.sort_values("ratio", ascending=False)
            self.features_adversarial = df_adv.copy()
            self.eval_adversarial_se  = se_eval.copy()
            self.eval_adversarial_df  = df_pred.copy()
        else:
            try: df_adv = getattr(self, "features_adversarial").copy()
            except AttributeError:
                self.logger.warning(f"features_adversarial is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        if cutoff is not None:
            if thre_count == "mean":
                df_adv = df_adv.loc[df_adv["count"] >= int(df_adv["count"].mean())]
            else:
                df_adv = df_adv.loc[df_adv["count"] >= thre_count]
            columns_del = df_adv.index[df_adv["ratio"] >= cutoff]
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del: self.logger.info(f"feature: {x}, ratio: {df_adv.loc[x, 'ratio']}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            assert len(columns) > 0, f"Increase cutoff: {cutoff} to keep at least one feature."
            self.update_features(columns)
        else:
            self.logger.info(f"cutoff is not set so no updated columns.")
        self.logger.info("END", color=["GREEN", "BOLD"])

    def cut_features_auto(
        self, df: pd.DataFrame=None, df_test: pd.DataFrame=None, 
        list_proc: list[str] = [
            "self.cut_features_by_variance(df, cutoff=0.99, ignore_nan=False)",
            "self.cut_features_by_variance(df, cutoff=0.99, ignore_nan=True )",
            "self.cut_features_by_randomtree_importance(df, cutoff=None, max_iter=5, min_count=1000, dtype=np.float32, batch_size=25)",
            "self.cut_features_by_adversarial_validation(df, df_test, cutoff=None, thre_count='mean', n_split=3, n_cv=2, dtype=np.float32, batch_size=25)",
            "self.cut_features_by_correlation(df, cutoff=0.99, dtype='float16', is_gpu=True, corr_type='pearson',  batch_size=2000, min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='spearman', batch_size=500,  min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='chatterjee', batch_size=500,  min_n=100)",
            # "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='kendall',  batch_size=500,  min_n=50, n_sample=250, n_iter=2)",
        ]
    ) -> typing.Self:
        self.logger.info("START", color=["GREEN", "BOLD"])
        for proc in list_proc:
            _out = eval(proc, {}, {"self": self, "df": df, "df_train": df, "df_test": df_test, "np": np, "pd": pd})
            if _out is not None:
                self.logger.info(f"output: {_out}")
        self.logger.info("END", color=["GREEN", "BOLD"])
        return self
    
    def proc_registry(self, dict_proc: dict | None=None, is_auto_colslct: bool=True):
        self.logger.info("START")
        assert isinstance(is_auto_colslct, bool)
        if dict_proc is None:
            if self.is_reg:
                dict_proc = {
                    "row": [
                        '"ProcDropNa", self.columns_ans[0]',
                    ],
                    "exp": [
                        '"ProcAsType", pl.Float32', 
                        '"ProcToValues"', 
                        '"ProcReplaceInf", posinf=float("nan"), neginf=float("nan")', 
                    ],
                    "ans": [
                        '"ProcAsType", pl.Float32',
                        '"ProcToValues"',
                        '"ProcReshape", (-1, )',
                    ]
                }
            else:
                dict_proc = {
                    "row": [
                        '"ProcDropNa", self.columns_ans[0]',
                    ],
                    "exp": [
                        '"ProcAsType", pl.Float32', 
                        '"ProcToValues"', 
                        '"ProcReplaceInf", posinf=float("nan"), neginf=float("nan")', 
                    ],
                    "ans": [
                        '"ProcAsType", pl.Int32',
                        '"ProcMapLabelAuto", self.columns_ans[0]',
                        '"ProcToValues"',
                        '"ProcReshape", (-1, )',
                        '"ProcAsType", np.int32',
                    ]
                }
        assert isinstance(dict_proc, dict)
        for x, y in dict_proc.items():
            assert x in ["row", "exp", "ans"]
            assert check_type_list(y, str)
        self.proc_row = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=False)
        self.proc_exp = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=is_auto_colslct)
        self.proc_ans = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=True)
        for _type in ["row", "exp", "ans"]:
            if _type in dict_proc:
                for x in dict_proc[_type]:
                    eval(f"self.proc_{_type}.register({x})", {}, {"self": self, "np": np, "pd": pd, "pl": pl})
        self.proc_check_init()
        self.logger.info("END")
    
    def proc_fit(self, df: DATAFRAME, is_row: bool=True, is_exp: bool=True, is_ans: bool=True):
        self.logger.info("START")
        assert isinstance(df, DATAFRAME)
        assert isinstance(is_row, bool)
        assert isinstance(is_exp, bool)
        assert isinstance(is_ans, bool)
        self.logger.info(f"proc fit row: {is_row}")
        if is_row:
            df, indexes = self.proc_row.fit(df, check_inout=["class"], is_return_index=True)
        else:
            indexes = df.index.copy() if isinstance(df, pd.DataFrame) else np.arange(df.shape[0], dtype=int)
        self.logger.info(f"df shape: {df.shape}")
        if is_exp == False and is_ans == False:
            return df, None, indexes
        self.logger.info(f"proc fit exp: {is_exp}")
        output_x = self.proc_exp.fit(df[self.columns],     check_inout=["row"], is_return_index=False) if is_exp else None
        self.logger.info(f"output_x shape: {output_x.shape if output_x is not None else None}")
        self.logger.info(f"proc fit ans: {is_ans}")
        output_y = self.proc_ans.fit(df[self.columns_ans], check_inout=["row"], is_return_index=False) if is_ans else None
        self.logger.info(f"output_y shape: {output_y.shape if output_y is not None else None}")
        self.logger.info("END")
        return output_x, output_y, indexes
    
    def proc_call(self, df: pd.DataFrame, is_row: bool=False, is_exp: bool=True, is_ans: bool=False):
        self.logger.info("START")
        assert isinstance(is_row, bool)
        assert isinstance(is_exp, bool)
        assert isinstance(is_ans, bool)
        self.logger.info(f"proc call row: {is_row}")
        if is_row:
            self.proc_row.check_proc(False)
            df, indexes = self.proc_row(df, is_return_index=True)
            self.proc_row.check_proc(True)
        else:
            indexes = df.index.copy() if isinstance(df, pd.DataFrame) else np.arange(df.shape[0], dtype=int)
        self.logger.info(f"df shape: {df.shape}")
        if is_exp == False and is_ans == False:
            return df, None, indexes
        self.logger.info(f"proc call exp: {is_exp}")
        output_x = self.proc_exp(df, is_return_index=False) if is_exp else None
        self.logger.info(f"output_x shape: {output_x.shape if output_x is not None else None}")
        self.logger.info(f"proc call ans: {is_ans}")
        output_y = self.proc_ans(df, is_return_index=False) if is_ans else None
        self.logger.info(f"output_y shape: {output_y.shape if output_y is not None else None}")
        self.logger.info("END")
        return output_x, output_y, indexes

    def proc_check_init(self):
        self.logger.info("START")
        self.proc_row.check_proc(True)
        self.proc_exp.check_proc(True)
        self.proc_ans.check_proc(True)
        self.logger.info("END")
    
    def predict(
        self, df: DATAFRAME_NONE=None, input_x: np.ndarray | None=None,
        is_row: bool=False, is_exp: bool=True, is_ans: bool=False, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.logger.info("START", color=["GREEN", "BOLD"])
        self.logger.info(f"kwargs: {kwargs}")
        assert is_exp
        if input_x is None:
            assert isinstance(df, DATAFRAME)
            input_x, input_y, input_index = self.proc_call(df, is_row=is_row, is_exp=is_exp, is_ans=is_ans)
        else:
            input_y, input_index = None, None
        output = getattr(self.get_model(), self.model_func)(input_x, **kwargs)
        self.logger.info("END", color=["GREEN", "BOLD"])
        return output, input_y, input_index
    
    def set_post_model(
        self, is_cv: bool=False, is_loop: bool=False,
        list_cv: list[int] | None=None, list_loop: list[int] | None=None,
        calibmodel: int | str | None=None, is_calib_after_cv: bool=False,
        is_normalize: bool=False, n_bins: int=None, df_calib: DATAFRAME_NONE=None, useerr: bool=True,
        kwargs_calib: dict={},
    ) -> typing.Self:
        self.logger.info("START", color=["GREEN", "BOLD"])
        self.logger.info(
            f"is_cv={is_cv}, calibmodel={calibmodel}, is_calib_after_cv={is_calib_after_cv}, " + 
            f"list_cv={list_cv}, list_loop={list_loop}, " + 
            f"is_normalize={is_normalize}, n_bins={n_bins}, df_calib={df_calib}, useerr={useerr}, " + 
            f"kwargs_calib={kwargs_calib}"
        )
        assert isinstance(is_cv, bool)
        assert isinstance(is_loop, bool)
        assert not (is_cv and is_loop)
        assert isinstance(calibmodel, (int, type(None)))
        is_calib = (calibmodel is not None)
        assert isinstance(is_calib_after_cv, bool)
        assert isinstance(is_normalize, bool)
        assert isinstance(useerr, bool)
        assert isinstance(n_bins, (int, type(None)))
        assert isinstance(df_calib, DATAFRAME_NONE)
        if not is_calib:
            assert df_calib is None
        assert isinstance(kwargs_calib, dict)
        if is_calib_after_cv: assert is_calib
        if is_cv:
            assert is_loop == False
            assert list_loop is None
            assert len(self.list_cv) > 0 and check_type_list(self.list_cv, str)
            if list_cv is not None:
                assert check_type_list(list_cv, int)
                assert all([(x > 0 and x <= len(self.list_cv)) for x in list_cv])
                list_cv = [str(x).zfill(len(self.list_cv[0])) for x in list_cv]
                assert all([x in self.list_cv for x in list_cv])
            else:
                list_cv = self.list_cv
            assert all([hasattr(self, f"model_cv{x}") for x in list_cv])
        elif is_loop:
            assert is_cv == False
            assert list_cv is None
            assert is_calib_after_cv == False
            assert len(self.list_loop) > 0 and check_type_list(self.list_loop, str)
            if list_loop is not None:
                assert check_type_list(list_loop, int)
                assert all([(x > 0 and x <= len(self.list_loop)) for x in list_loop])
                list_loop = [str(x).zfill(len(self.list_loop[0])) for x in list_loop]
                assert all([x in self.list_loop for x in list_loop])
            else:
                list_loop = self.list_loop
            assert all([hasattr(self, f"model_loop{x}") for x in list_loop])
        else:
            ## is_cv == False, is_loop == False
            assert list_cv   is None
            assert list_loop is None
            assert is_calib
        # set post model
        if is_cv:
            if is_calib_after_cv:
                assert is_calib
                modelcv = MultiModel([getattr(self, f"model_cv{x}") for x in list_cv], func_predict=self.model_func)
                modelcalib = Calibrator(
                    modelcv, self.model_func, is_normalize=is_normalize, is_reg=self.is_reg, 
                    calibmodel=calibmodel, useerr=useerr, **kwargs_calib
                )
                if modelcalib.calibrator.is_fit_required:
                    assert df_calib is not None
                    input_x, input_y, _ = self.proc_call(df_calib, is_row=True, is_exp=True, is_ans=True)
                    modelcalib.fit(input_x, input_y, is_input_prob=False, n_bins=n_bins)
                self.model_post = modelcalib
            else:
                if is_calib:
                    assert df_calib is None, "validation data is used so no need extra dataframe for calibration."
                    models = []
                    for x in list_cv:
                        self.logger.info(f"calibration for {x} ...")
                        modelcalib = Calibrator(
                            getattr(self, f"model_cv{x}"), self.model_func, is_normalize=is_normalize, 
                            calibmodel=calibmodel, useerr=useerr, **kwargs_calib
                        )
                        if modelcalib.calibrator.is_fit_required:
                            df         = getattr(self, f"eval_valid_df_cv{x}").copy()
                            input_x    = df.loc[:, df.columns.str.contains("^predict_proba_", regex=True)].to_numpy(dtype=float)
                            input_y    = df.loc[:, df.columns == "answer"].to_numpy().reshape(-1).astype(int)
                            assert input_x.ndim == 2
                            assert input_y.ndim == 1
                            modelcalib.fit(input_x, input_y, is_input_prob=True, n_bins=n_bins)
                        models.append(modelcalib)
                    self.model_post = MultiModel(models, func_predict=self.model_func)
                else:
                    self.model_post = MultiModel([getattr(self, f"model_cv{x}") for x in list_cv], func_predict=self.model_func)
        elif is_loop:
            modelloop = MultiModel([getattr(self, f"model_loop{x}") for x in list_loop], func_predict=self.model_func)
            if is_calib:
                modelcalib = Calibrator(
                    modelloop, self.model_func, is_normalize=is_normalize, is_reg=self.is_reg, 
                    calibmodel=calibmodel, useerr=useerr, **kwargs_calib
                )
                if modelcalib.calibrator.is_fit_required:
                    assert df_calib is not None
                    input_x, input_y, _ = self.proc_call(df_calib, is_row=True, is_exp=True, is_ans=True)
                    modelcalib.fit(input_x, input_y, is_input_prob=False, n_bins=n_bins)
                self.model_post = modelcalib
            else:
                self.model_post = modelloop
        else:
            assert is_calib
            modelcalib = Calibrator(
                self.model, self.model_func, is_normalize=is_normalize, is_reg=self.is_reg, 
                calibmodel=calibmodel, useerr=useerr, **kwargs_calib
            )
            if modelcalib.calibrator.is_fit_required:
                assert df_calib is not None
                input_x, input_y, _ = self.proc_call(df_calib, is_row=True, is_exp=True, is_ans=True)
                modelcalib.fit(input_x, input_y, is_input_prob=False, n_bins=n_bins)
            self.model_post = modelcalib
        self.is_postmodel = True
        self.logger.info(self.model_post.to_json(mode=2, indent=4))
        self.logger.info("END", color=["GREEN", "BOLD"])
        return self

    def get_model_mode(self):
        if self.is_postmodel:
            model_mode = "model_post"
        else:
            model_mode = "model"
        return model_mode

    def get_model(self):
        model_mode = self.get_model_mode()
        self.logger.info(f"model mode: {model_mode}")
        return getattr(self, model_mode)

    def fit(
        self, df_train: DATAFRAME, df_valid: DATAFRAME_NONE=None, is_proc_fit: bool=True, 
        params_fit: str | dict={}, params_fit_evaldict: dict={}, is_eval_train: bool=False, dict_extra_cols: dict[str, str]=None,
    ):
        """
        params_fit:
            Below are preserved keys. "_train_XXXXX" is basically used for sample weight.
            "_validation_x", "_validation_y", "_train_XXXXX", "_valid_XXXXX", "_random_seed"
        """
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_train, DATAFRAME)
        assert isinstance(df_valid, DATAFRAME_NONE)
        for x in self.columns_oth:
            assert np.any(np.array(df_train.columns, dtype=object) == x)
            if df_valid is not None: assert np.any(np.array(df_valid.columns, dtype=object) == x)
        assert isinstance(is_proc_fit, bool)
        assert isinstance(params_fit, (str, dict))
        assert isinstance(is_eval_train, bool)
        if dict_extra_cols is not None:
            assert isinstance(dict_extra_cols, dict)
            for x in dict_extra_cols.keys(): assert x in df_train.columns
        # pre proc
        if is_proc_fit:
            train_x, train_y, train_index = self.proc_fit(df_train, is_row=True, is_exp=True, is_ans=True)
        else:
            train_x, train_y, train_index = self.proc_call(df_train, is_row=True, is_exp=True, is_ans=True)
        valid_x, valid_y = None, None
        if df_valid is not None:
            if isinstance(params_fit, str):
                if (params_fit.find("_validation_x") < 0) or (params_fit.find("_validation_y") < 0):
                    self.logger.warning("You set the validation data but the data won't pass to the model.")
            else:
                if (not unmask_value_isin_object(params_fit, ["_validation_x"])) or (not unmask_value_isin_object(params_fit, ["_validation_y"])):
                    self.logger.warning("You set the validation data but the data won't pass to the model.")
            valid_x, valid_y, valid_index = self.proc_call(df_valid, is_row=True, is_exp=True, is_ans=True)
        if self.model is None: self.logger.raise_error("model is not set.")
        # other columns (like sample weight)
        dict_extra = {"_validation_x": valid_x, "_validation_y": valid_y, "_random_seed": self.random_seed}
        if dict_extra_cols is not None:
            for x, y in dict_extra_cols.items():
                if isinstance(df_train, pl.DataFrame):
                    dict_extra[f"_train_{y}"] = df_train[train_index, x].to_numpy()
                    if df_valid is not None:
                        dict_extra[f"_valid_{y}"] = df_valid[valid_index, x].to_numpy()
                else:
                    dict_extra[f"_train_{y}"] = df_train.loc[train_index, x].to_numpy().copy()
                    if df_valid is not None:
                        dict_extra[f"_valid_{y}"] = df_valid.loc[valid_index, x].to_numpy().copy()
                self.logger.info(f"created extra parameter: _train_{y}")
                if df_valid is not None:
                    self.logger.info(f"created extra parameter: _valid_{y}")
        # update params_fit_evaldict
        assert isinstance(params_fit_evaldict, dict)
        params_fit_evaldict = copy.deepcopy(params_fit_evaldict) | dict_extra
        # update params_fit
        if isinstance(params_fit, str):
            params_fit = eval(params_fit, {}, params_fit_evaldict)
        else:
            params_fit = copy.deepcopy(params_fit)
            params_fit = unmask_values(params_fit, params_fit_evaldict)
        self.logger.info(f"model: {self.model}, is_reg: {self.is_reg}, fit params: {params_fit}")
        self.logger.info("Fitting START ...", color=["CYAN"])
        self.model.fit(train_x, train_y, **params_fit)
        self.logger.info("Fitting END !!!", color=["CYAN"])
        self.is_fit = True
        # eval
        if valid_x is not None:
            self.logger.info("Evaluate valid START ...", color=["CYAN"])
            se_eval, df_eval = eval_model(valid_x, valid_y, model=self.model, func_predict=self.model_func, is_reg=self.is_reg)
            self.eval_valid_se = se_eval.copy()
            self.eval_valid_df = df_eval.copy()
            self.eval_valid_df["index"] = valid_index
            for x in self.columns_oth:
                if isinstance(df_valid, pl.DataFrame):
                    ndf = df_valid[valid_index, x].to_numpy()
                else:
                    ndf = df_valid.loc[valid_index, x].to_numpy()
                self.eval_valid_df[f"oth_{x}"] = ndf
            for x in se_eval.index:
                self.logger.info(f"{x}: {se_eval.loc[x]}")
            self.logger.info("Evaluate valid END !!!", color=["CYAN"])
        if is_eval_train:
            self.logger.info("Evaluate train START ...", color=["CYAN"])
            se_eval, df_eval = eval_model(train_x, train_y, model=self.model, func_predict=self.model_func, is_reg=self.is_reg)
            self.eval_train_se = se_eval.copy()
            self.eval_train_df = df_eval.copy()
            self.eval_train_df["index"] = train_index
            for x in self.columns_oth:
                if isinstance(df_train, pl.DataFrame):
                    ndf = df_train[train_index, x].to_numpy()
                else:
                    ndf = df_train.loc[train_index, x].to_numpy()
                self.eval_train_df[f"oth_{x}"] = ndf
            for x in se_eval.index:
                self.logger.info(f"{x}: {se_eval.loc[x]}")
            self.logger.info("Evaluate train END !!!", color=["CYAN"])
        else:
            self.eval_train_se = pd.Series({"n_data": train_x.shape[0]}, dtype=object)
        self.logger.info("END", color=["GREEN", "BOLD"])

    def fit_cross_validation(
        self, df_train: DATAFRAME,
        n_split: int | None=None, n_cv: int | None=None, mask_split: np.ndarray | None=None,
        cols_multilabel_split: list[str] | None=None, group_split: str | list[str] | None=None,
        indexes_train: list[np.ndarray] | None=None, indexes_valid: list[np.ndarray] | None=None,
        params_fit: str | dict={}, params_fit_evaldict: dict={},
        is_proc_fit_every_cv: bool=True, is_save_cv_models: bool=False, dict_extra_cols: dict[str, str] | None=None,
    ):
        """
        n_split:
            The number of splits for the data
        n_cv:
            The number of cross_validation
        mask_split:
            This is for mask from split, it means thah masked data must be used in all validations
        cols_multilabel_split:
            Basically, answer column is used for splting
            if you want to split by multi columns, use this option
        params_fit:
            This is used like ...
            >>> model.fit(train_x, train_y, **params_fit)'
        params_fit_evaldict:
            This is used like ...
            >>> params_fit_evaldict.update({"_validation_x": valid_x, "_validation_y": valid_y})
            >>> params_fit = eval(params_fit, {}, params_fit_evaldict)
        """
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_train, DATAFRAME)
        assert isinstance(is_proc_fit_every_cv, bool)
        assert isinstance(is_save_cv_models, bool)
        if n_split is None:
            assert indexes_train is not None and check_type_list(indexes_train, np.ndarray)
            assert indexes_valid is not None and check_type_list(indexes_valid, np.ndarray)
            assert len(indexes_train) == len(indexes_valid)
            n_cv = len(indexes_train) if n_cv is None else n_cv
        else:
            assert isinstance(n_split, int) and n_split >= 2
            assert isinstance(n_cv,    int) and n_cv    >= 1 and n_cv <= n_split
            assert indexes_train is None
            assert indexes_valid is None
        if mask_split is not None:
            assert isinstance(mask_split, np.ndarray)
            assert len(mask_split.shape) == 1
            assert df_train.shape[0] == mask_split.shape[0]
            assert mask_split.dtype in [bool, np.bool_]
            assert group_split is None # group_split is not used in mask_split
        if cols_multilabel_split is not None:
            if isinstance(cols_multilabel_split, str): cols_multilabel_split = [cols_multilabel_split]
            assert isinstance(cols_multilabel_split, list)
            assert check_type_list(cols_multilabel_split, str)
        if group_split is not None:
            if isinstance(group_split, str): group_split = [group_split]
            assert check_type_list(group_split, str)
            for x in group_split: assert x in df_train.columns
        # proc fitting
        index_org = df_train.index.to_numpy() if isinstance(df_train, pd.DataFrame) else np.arange(df_train.shape[0], dtype=int)
        df_train, _, _indexes = self.proc_fit(df_train, is_row=True, is_exp=False, is_ans=False)
        if not is_proc_fit_every_cv:
            self.proc_fit(df_train, is_row=True, is_exp=True, is_ans=True)
        is_fit = (not is_proc_fit_every_cv)
        # update mask split
        if mask_split is not None:
            if isinstance(df_train, pd.DataFrame):
                ndf_bool = index_org.isin(_indexes)
            else:
                ndf_bool = np.isin(index_org, _indexes)
            mask_split = mask_split[ndf_bool]
        # split for validation
        if n_split is not None:
            if group_split is not None:
                if isinstance(df_train, pl.DataFrame):
                    df_grp = df_train[group_split].clone().to_pandas().reset_index(drop=True)
                else:
                    df_grp = df_train.loc[:, group_split].copy().reset_index(drop=True)
                ndf_grp = df_grp.groupby(group_split)[df_grp.columns[0]].apply(lambda x: x.index.tolist()).to_numpy()
            indexes_train, indexes_valid = [], []
            if is_fit: _, ndf_y, _ = self.proc_call(df_train, is_row=False, is_exp=False, is_ans=True)
            else:      _, ndf_y, _ = self.proc_fit( df_train, is_row=False, is_exp=False, is_ans=True)
            indexes     = np.arange(df_train.shape[0], dtype=int)
            if cols_multilabel_split is None:
                self.logger.info(f"Use splitter: StratifiedKFold, n_split: {n_split}")
                splitter = StratifiedKFold(n_splits=n_split)
            else:
                self.logger.info(f"Use splitter: MultilabelStratifiedKFold, n_split: {n_split}, cols_multilabel_split: {cols_multilabel_split}")
                ndf_oth = df_train[cols_multilabel_split].to_numpy()
                splitter = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
                if len(ndf_y.shape) == 1: ndf_y = ndf_y.reshape(-1, 1)
                ndf_y = np.concatenate([ndf_oth, ndf_y], axis=1)
            if mask_split is not None:
                indexes_mask = indexes[ mask_split].copy()
                indexes      = indexes[~mask_split].copy()
                ndf_y        = ndf_y[~mask_split]
                self.logger.info(f"Use mask split. mask indexes: {indexes_mask}")
            try:
                if group_split is None:
                    generatot   = splitter.split(indexes, ndf_y)
                else:
                    indexes_grp = [x[0] for x in ndf_grp]
                    generatot   = splitter.split(np.arange(len(ndf_grp), dtype=int), ndf_y[indexes_grp])
                for i_split, (index_train, index_valid) in enumerate(generatot):
                    if mask_split is not None:
                        index_train = np.append(index_train, indexes_mask.copy())
                    if group_split is not None:
                        assert mask_split is None
                        index_train = np.concatenate(ndf_grp[index_train], axis=0)
                        index_valid = np.concatenate(ndf_grp[index_valid], axis=0)
                    self.logger.info(f"Split: {i_split}. \ntrain indexes: {index_train}\nvalid indexes: {index_valid}")
                    indexes_train.append(index_train)
                    indexes_valid.append(index_valid)
            except ValueError as e:
                self.logger.warning(f"{e}")
                self.logger.warning("use normal random splitter. 'mask_split' & 'cols_multilabel_split' functions are ignored.")
                indexes     = np.arange(df_train.shape[0], dtype=int)
                index_split = np.array_split(indexes, n_split)
                for i in range(n_cv):
                    indexes_train.append(indexes[~np.isin(indexes, index_split[i])].copy())
                    indexes_valid.append(index_split[i].copy())
        # cross validation
        self.logger.info(f"params_fit: {params_fit}, params_fit_evaldict: {params_fit_evaldict}, dict_extra_cols: {dict_extra_cols}")
        for i_cv, (index_train, index_valid) in enumerate(zip(indexes_train, indexes_valid)):
            i_cv += 1
            self.logger.info(f"cross validation : {i_cv} / {n_cv} start...", color=["CYAN"])
            self.reset_model()
            self.fit(
                df_train=df_train.iloc[index_train, :] if isinstance(df_train, pd.DataFrame) else df_train[index_train],
                df_valid=df_train.iloc[index_valid, :] if isinstance(df_train, pd.DataFrame) else df_train[index_valid],
                is_proc_fit=is_proc_fit_every_cv, params_fit=params_fit, params_fit_evaldict=params_fit_evaldict,
                is_eval_train=False, dict_extra_cols=dict_extra_cols
            )
            self.logger.info(f"cross validation : {i_cv} / {n_cv} end  ...", color=["CYAN"])
            setattr(self, f"eval_valid_df_cv{str(i_cv).zfill(len(str(n_cv)))}", self.eval_valid_df)
            setattr(self, f"eval_valid_se_cv{str(i_cv).zfill(len(str(n_cv)))}", self.eval_valid_se)
            if is_save_cv_models:
                if hasattr(self.model, "copy"):
                    setattr(self, f"model_cv{str(i_cv).zfill(len(str(n_cv)))}", self.model.copy())
                else:
                    setattr(self, f"model_cv{str(i_cv).zfill(len(str(n_cv)))}", copy.deepcopy(self.model))
            if i_cv >= n_cv: break
        self.list_cv = [f"{str(i_cv+1).zfill(len(str(n_cv)))}" for i_cv in range(n_cv)]
        self.logger.info("END", color=["GREEN", "BOLD"])
    
    def fit_loop(
        self, df_train: DATAFRAME, df_valid: DATAFRAME_NONE=None, n_loop: int=2, 
        params_fit: str | dict={}, params_fit_evaldict: dict={},
        is_save_models: bool=True, dict_extra_cols: dict[str, str] | None=None,
    ):
        """
        n_loop:
            The number of loop
        """
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_train, DATAFRAME)
        assert isinstance(df_valid, DATAFRAME_NONE)
        for x in self.columns_oth:
            assert np.any(np.array(df_train.columns, dtype=object) == x)
            if df_valid is not None: assert np.any(np.array(df_valid.columns, dtype=object) == x)
        assert isinstance(n_loop, int) and n_loop >= 1
        assert isinstance(params_fit, (str, dict))
        assert isinstance(is_save_models, bool)
        if dict_extra_cols is not None:
            assert isinstance(dict_extra_cols, dict)
            for x in dict_extra_cols.keys(): assert x in df_train.columns
        assert unmask_value_isin_object(params_fit, ["_random_seed"])
        # proc fitting
        self.proc_fit(df_train, is_row=True, is_exp=True, is_ans=True)
        # fit loop
        org_random_seed = self.random_seed
        for i_loop in range(n_loop):
            i_loop += 1
            self.logger.info(f"fit loop : {i_loop} / {n_loop} start...", color=["CYAN"])
            self.random_seed = org_random_seed + i_loop
            self.reset_model()
            self.fit(
                df_train=df_train, df_valid=df_valid,
                is_proc_fit=False, params_fit=params_fit, params_fit_evaldict=params_fit_evaldict,
                is_eval_train=False, dict_extra_cols=dict_extra_cols
            )
            self.logger.info(f"fit loop : {i_loop} / {n_loop} end...", color=["CYAN"])
            setattr(self, f"eval_valid_df_loop{str(i_loop).zfill(len(str(n_loop)))}", self.eval_valid_df)
            setattr(self, f"eval_valid_se_loop{str(i_loop).zfill(len(str(n_loop)))}", self.eval_valid_se)
            if is_save_models:
                if hasattr(self.model, "copy"):
                    setattr(self, f"model_loop{str(i_loop).zfill(len(str(n_loop)))}", self.model.copy())
                else:
                    setattr(self, f"model_loop{str(i_loop).zfill(len(str(n_loop)))}", copy.deepcopy(self.model))
        self.random_seed = org_random_seed
        self.list_loop = [f"{str(i_loop+1).zfill(len(str(n_loop)))}" for i_loop in range(n_loop)]
        self.logger.info("END", color=["GREEN", "BOLD"])

    def fit_basic_treemodel(
        self, df_train: DATAFRAME, df_valid: DATAFRAME_NONE=None, df_test: DATAFRAME_NONE=None,
        ncv: int=2, n_estimators: int=100, model_kwargs: dict={}
    ):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_train, DATAFRAME)
        assert isinstance(df_valid, DATAFRAME_NONE)
        assert isinstance(df_test,  DATAFRAME_NONE)
        assert isinstance(ncv,          int) and ncv >= 1
        assert isinstance(n_estimators, int) and n_estimators >= 2
        assert not (ncv == 1 and df_valid is None)
        assert not (ncv >= 2 and df_valid is not None)
        assert isinstance(model_kwargs, dict)
        # set model
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        dictwk = {
            "bootstrap": True, "n_estimators": n_estimators, "max_depth": None, "max_features":"sqrt", 
            "verbose":3, "random_state": 0, "n_jobs": self.n_jobs
        }
        dictwk.update(model_kwargs)
        if self.is_reg: self.set_model(RandomForestRegressor,  **dictwk)
        else:           self.set_model(RandomForestClassifier, class_weight="balanced", **dictwk)
        # registry proc
        self.proc_registry()
        self.proc_exp.register("ProcFillNaMinMaxRandomly")
        self.proc_exp.register("ProcFillNa", 0)
        # training
        if df_valid is not None and ncv == 1:
            self.fit(df_train, df_valid=df_valid, is_proc_fit=True, is_eval_train=True)
        else:
            self.fit_cross_validation(df_train, n_split=ncv, n_cv=ncv, is_proc_fit_every_cv=True, is_save_cv_models=True)
            self.set_post_model(is_cv=True)
        # test evaluation
        if df_test is not None:
            self.evaluate(df_test, is_store=True)
        self.logger.info("END", color=["GREEN", "BOLD"])
    
    def evaluate(self, df_test: DATAFRAME, columns_ans: str | np.ndarray | None=None, is_store: bool=False, **kwargs):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(df_test, DATAFRAME)
        if columns_ans is not None:
            assert isinstance(columns_ans, (str, np.ndarray))
            if isinstance(columns_ans, str):
                assert columns_ans in df_test.columns
            else:
                assert columns_ans.shape[0] == df_test.shape[0]
        assert isinstance(is_store, bool)
        if isinstance(columns_ans, np.ndarray):
            test_y = columns_ans.copy()
        elif isinstance(columns_ans, str):
            test_y = df_test[columns_ans].to_numpy()
        if columns_ans is None:
            test_x, test_y, test_index = self.proc_call(df_test, is_row=True, is_exp=True, is_ans=True)
        else:
            test_x, _,      test_index = self.proc_call(df_test, is_row=True, is_exp=True, is_ans=False)
            test_y = test_y[test_index]
        se_eval, df_eval = eval_model(test_x, test_y, model=self.get_model(), is_reg=self.is_reg, func_predict=self.model_func, **kwargs)
        for x in se_eval.index:
            self.logger.info(f"{x}: {se_eval.loc[x]}")
        if is_store:
            self.eval_test_se = se_eval.copy()
            self.eval_test_df = df_eval.copy()
            self.eval_test_df["index"] = test_index
        self.logger.info("END", color=["GREEN", "BOLD"])
        return se_eval, df_eval
    
    def re_evalate(self):
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert self.is_fit, "You should fit the model first"
        # cross validation
        for icv in self.list_cv:
            self.logger.info(f"re-evaluate cross validation: {icv}")
            dfwk    = getattr(self, f"eval_valid_df_cv{icv}").copy()
            valid_x = dfwk.loc[:, dfwk.columns.str.contains("predict_proba_")].to_numpy()
            valid_y = dfwk["answer"].to_numpy()
            se_eval, _ = eval_model(valid_x, valid_y, model=None, func_predict=None, is_reg=self.is_reg)
            setattr(self, f"eval_valid_se_cv{icv}", se_eval.copy())
        # fit loop
        for iloop in self.list_loop:
            self.logger.info(f"re-evaluate fit loop: {iloop}")
            dfwk    = getattr(self, f"eval_valid_df_loop{iloop}").copy()
            valid_x = dfwk.loc[:, dfwk.columns.str.contains("predict_proba_")].to_numpy()
            valid_y = dfwk["answer"].to_numpy()
            se_eval, _ = eval_model(valid_x, valid_y, model=None, func_predict=None, is_reg=self.is_reg)
            setattr(self, f"eval_valid_se_loop{iloop}", se_eval.copy())
        # validation
        dfwk = self.eval_valid_df.copy()
        if dfwk.shape[0] > 0:
            self.logger.info(f"re-evaluate validation")
            valid_x = dfwk.loc[:, dfwk.columns.str.contains("predict_proba_")].to_numpy()
            valid_y = dfwk["answer"].to_numpy()
            se_eval, _ = eval_model(valid_x, valid_y, model=None, func_predict=None, is_reg=self.is_reg)
            self.eval_valid_se = se_eval.copy()
        # test
        dfwk = self.eval_test_df.copy()
        if dfwk.shape[0] > 0:
            self.logger.info(f"re-evaluate test")
            valid_x = dfwk.loc[:, dfwk.columns.str.contains("predict_proba_")].to_numpy()
            valid_y = dfwk["answer"].to_numpy()
            se_eval, _ = eval_model(valid_x, valid_y, model=None, func_predict=None, is_reg=self.is_reg)
            self.eval_test_se = se_eval.copy()
        self.logger.info("END", color=["GREEN", "BOLD"])

    def set_n_jobs(self, n_jobs: int):
        self.n_jobs          = n_jobs
        self.proc_row.n_jobs = n_jobs
        self.proc_exp.n_jobs = n_jobs
        self.proc_ans.n_jobs = n_jobs
    
    def write_log(self, filepath: str, encoding: str="utf8"):
        assert isinstance(filepath, str)
        assert isinstance(encoding, str)
        with open(filepath, mode='w', encoding=encoding) as f:
            f.write(self.logger.internal_stream.getvalue())

    def save(
        self, dirpath: str=f"__tmp__", filename: str=None, is_remake: bool=False, is_minimum: bool=False, 
        is_json: bool=False, mode: int=0, encoding: str="utf8"
    ):
        """
        mode:
            0: base64 encoding
            1: save to file
            2: only class name (no save object)
        """
        self.logger.info("START", color=["GREEN", "BOLD"])
        assert isinstance(dirpath, str) or dirpath is None
        assert isinstance(is_remake, bool)
        assert isinstance(is_minimum, bool)
        assert isinstance(is_json,   bool)
        assert isinstance(mode,      int) and mode in [0, 1, 2]
        assert isinstance(encoding,  str)
        dirpath = correct_dirpath(dirpath) if dirpath is not None else "./"
        assert not (dirpath == "./" and is_remake == True)
        makedirs(dirpath, exist_ok=True, remake=is_remake)
        filename = f"mlmanager.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.{id(self)}" if filename is None else filename
        ins      = self.copy(is_minimum=is_minimum) if is_minimum else self
        ins.logger.info(f"save file: {dirpath + filename}")
        if is_json:
            fname = (dirpath + filename + ".min.json") if is_minimum else (dirpath + filename + ".json")
            if mode == 1: makedirs(fname.replace(".json", "") + "/", exist_ok=True, remake=True)
            with open(fname, mode='w', encoding=encoding) as f:
                f.write(ins.to_json(indent=4, mode=mode, savedir=fname.replace(".json", "") + "/"))
            ins.write_log(fname + ".log", encoding=encoding)
        else:
            fname = (dirpath + filename + ".min.pickle") if is_minimum else (dirpath + filename + ".pickle")
            with open(fname, mode='wb') as f:
                pickle.dump(ins, f, protocol=PICKLE_PROTOCOL)
            ins.write_log(fname + ".log", encoding=encoding)
        self.logger.info("END", color=["GREEN", "BOLD"])

    @classmethod
    def load(cls, filepath: str, n_jobs: int, encoding: str="utf8"):
        LOGGER.info("START", color=["GREEN", "BOLD"])
        assert isinstance(n_jobs, int)
        assert isinstance(filepath, str)
        LOGGER.info(f"load file: {filepath}")
        if filepath.endswith(".json"):
            with open(filepath, mode='r', encoding=encoding) as f:
                dictwk = json.load(f)
            manager = cls.from_dict(dictwk, n_jobs=n_jobs, basedir=filepath.replace(".json", "") + "/")
        else:
            with open(filepath, mode='rb') as f:
                manager: MLManager = pickle.load(f)
            manager.__class__ = MLManager
            manager.logger = set_logger(manager.logger.name, internal_log=True)
            manager.set_n_jobs(n_jobs)
            if os.path.exists(filepath + ".log"):
                LOGGER.info(f"load log file: {filepath + '.log'}")
                with open(filepath + ".log", mode='r') as f:
                    manager.logger.internal_stream.write(f.read())
            manager.logger.info(f"load: {filepath}, jobs: {n_jobs}")
        tmp = cls(manager.columns_exp.tolist(), manager.columns_ans.tolist())
        for x in dir(tmp):
            if not hasattr(manager, x):
                setattr(manager, x, getattr(tmp, x)) # initialize
        LOGGER.info("END", color=["GREEN", "BOLD"])
        return manager
