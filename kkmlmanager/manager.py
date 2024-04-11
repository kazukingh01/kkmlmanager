import sys, pickle, os, copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# local package
from kkmlmanager.regproc import RegistryProc
from kkmlmanager.features import get_features_by_variance, get_features_by_correlation, get_features_by_randomtree_importance, get_features_by_adversarial_validation
from kkmlmanager.eval import eval_model
from kkmlmanager.models import MultiModel
from kkmlmanager.calibration import Calibrater
from kkmlmanager.util.numpy import isin_compare_string
from kkmlmanager.util.com import check_type, check_type_list, correct_dirpath, makedirs
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "MLManager",
    "load_manager",
]


class MLManager:
    """
    Usage::
        see: https://github.com/kazukingh01/kkmlmanager/tree/main/tests
    """
    def __init__(
        self,
        # model parameter
        columns_exp: list[str], columns_ans: str | list[str], columns_otr: list[str]=None, is_reg: bool=False, 
        # common parameter
        outdir: str="./output/", random_seed: int=1, n_jobs: int=1
    ):
        self.logger = set_logger(f"{__class__.__name__}.{id(self)}", internal_log=True)
        self.logger.info("START")
        if isinstance(columns_ans, str): columns_ans = [columns_ans, ]
        if columns_otr is None: columns_otr = []
        assert check_type_list(columns_exp, str)
        assert check_type_list(columns_ans, str)
        assert check_type_list(columns_otr, str)
        assert isinstance(is_reg, bool)
        assert isinstance(outdir, str)
        assert isinstance(random_seed, int) and random_seed >= 0
        assert isinstance(n_jobs, int) and n_jobs >= 1
        self.columns_exp = np.array(columns_exp)
        self.columns_ans = np.array(columns_ans)
        self.columns_otr = np.array(columns_otr)
        self.is_reg      = is_reg
        self.outdir      = correct_dirpath(outdir)
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self.initialize()
        self.logger.info("END")
    
    def __str__(self):
        return f"model: {self.model}\ncolumns explain: {self.columns_exp}\ncolumns answer: {self.columns_ans}\ncolumns other: {self.columns_otr}"

    def initialize(self):
        self.logger.info("START")
        self.model        = None
        self.model_class  = None
        self.model_args   = None
        self.model_kwargs = None
        self.model_multi: MultiModel = None
        self.is_cvmodel   = False
        self.calibrater: Calibrater = None
        self.is_fit       = False
        self.is_calib     = False
        self.list_cv      = []
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
    
    def to_json(self, is_detail: bool=False):
        assert isinstance(is_detail, bool)
        return {
            "base": {
                "input":  self.columns.tolist() if is_detail else len(self.columns),
                "target": self.columns_ans.tolist(),
                "is_reg": self.is_reg,
                "random_seed": self.random_seed,
                "n_jobs": self.n_jobs,
            },
            "preproc": {
                "proc_row": str(self.proc_row),
                "proc_exp": str(self.proc_exp),
                "proc_ans": str(self.proc_ans),
            },
            "model": {
                "is_fit": self.is_fit,
                "is_calib": self.is_calib,
                "is_cvmodel": self.is_cvmodel,
                "mode": self.get_model_mode(),
                "cv": len(self.list_cv),
                "model": self.model.to_json() if self.model.__class__.__name__ == "KkGBDT" else str(self.model),
                "calibrater":  self.calibrater.to_json() if self.calibrater is not None else None,
                "model_multi": (
                    (self.model_multi.to_json() if self.model_multi is not None else {}) | 
                    ({"calib": self.model_multi.models[0].to_json()} if self.model_multi is not None and isinstance(self.model_multi.models[0], Calibrater) else {})
                ),
            },
            "eval": {
                "train": self.eval_train_se.to_dict(),
                "valid": [getattr(self, f"eval_valid_se_cv{x}").to_dict() for x in self.list_cv] if len(self.list_cv) > 0 else self.eval_valid_se.to_dict(),
                "test" : self.eval_test_se.to_dict(),
            }
        }

    def copy(self, is_minimum: bool=False):
        assert isinstance(is_minimum, bool)
        if is_minimum:
            ins = __class__(
                self.columns_exp.tolist(), self.columns_ans.tolist(), columns_otr=self.columns_otr.tolist(), 
                is_reg=self.is_reg, outdir=self.outdir, random_seed=self.random_seed, n_jobs=self.n_jobs
            )
            ins.logger.internal_stream.write(copy.deepcopy(self.logger.internal_stream.getvalue()))
            if hasattr(self.model, "copy"):
                ins.model    = self.model.copy()
            else:
                ins.model    = copy.deepcopy(self.model)
            ins.model_class  = copy.deepcopy(self.model_class)
            ins.model_args   = copy.deepcopy(self.model_args)
            ins.model_kwargs = copy.deepcopy(self.model_kwargs)
            ins.model_func   = copy.deepcopy(self.model_func)
            ins.columns      = self.columns.copy()
            ins.columns_hist = copy.deepcopy(self.columns_hist)
            ins.proc_row     = copy.deepcopy(self.proc_row)
            ins.proc_exp     = copy.deepcopy(self.proc_exp)
            ins.proc_ans     = copy.deepcopy(self.proc_ans)
            ins.is_fit       = self.is_fit
            ins.is_cvmodel   = self.is_cvmodel
            ins.is_calib     = self.is_calib
            ins.list_cv      = copy.deepcopy(self.list_cv)
            model_mode       = self.get_model_mode()
            if   model_mode == "model_multi":
                ins.model       = None
                ins.model_multi = copy.deepcopy(self.model_multi)
                ins.calibrater  = None
            elif model_mode == "calibrater":
                ins.model       = None
                ins.model_multi = None
                ins.calibrater  = copy.deepcopy(self.calibrater)
            return ins
        else:
            return copy.deepcopy(self)

    def set_model(self, model, *args, model_func_predict: str=None, **kwargs):
        self.logger.info("START")
        self.logger.info(f"model: {model}, args: {args}, model_func_predict: {model_func_predict}, kwargs: {kwargs}")
        self.model_class  = model
        self.model_args   = args
        self.model_kwargs = kwargs
        if model_func_predict is None:
            if self.is_reg: model_func_predict = "predict"
            else:           model_func_predict = "predict_proba"
        assert isinstance(model_func_predict, str)
        self.model_func   = model_func_predict
        self.reset_model()
        self.logger.info("END")
    
    def reset_model(self):
        self.logger.info("START")
        self.model       = self.model_class(*self.model_args, **self.model_kwargs)
        self.model_multi = None
        self.is_cvmodel  = False
        self.calibrater  = None
        self.is_fit      = False
        self.is_calib    = False
        self.logger.info("END")

    def update_features(self, features: list[str] | np.ndarray):
        self.logger.info("START")
        assert check_type(features, [np.ndarray, list])
        if isinstance(features, list):
            check_type_list(features, str)
            features = np.ndarray(features)
        assert np.all(isin_compare_string(features, self.columns))
        self.columns_hist.append(self.columns.copy())
        self.columns = features.copy()
        self.logger.info(f"columns new: {self.columns.shape}, columns before:{self.columns_hist[-1].shape}")
        self.logger.info("END")

    def cut_features_by_variance(self, df: pd.DataFrame=None, cutoff: float=0.99, ignore_nan: bool=False, batch_size: int=128):
        self.logger.info("START")
        assert df is None or isinstance(df, pd.DataFrame)
        assert isinstance(cutoff, float) and 0 < cutoff <= 1.0
        assert isinstance(ignore_nan, bool)
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, ignore_nan: {ignore_nan}")
        attr_name = f"features_var_{ignore_nan}_{str(cutoff)[:5].replace('.', '')}"
        if df is not None:
            sebool = get_features_by_variance(df[self.columns], cutoff=cutoff, ignore_nan=ignore_nan, batch_size=batch_size, n_jobs=self.n_jobs)
            setattr(self, attr_name, sebool.copy())
        else:
            try: sebool = getattr(self, attr_name)
            except AttributeError:
                self.logger.warning(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        columns = sebool.index[~sebool].values
        if df is not None:
            columns_del = self.columns[~isin_compare_string(self.columns, columns)].copy()
            for x in columns_del:
                sewk = df[x].value_counts().sort_values(ascending=False)
                self.logger.info(
                    f"feature: {x}, n nan: {df[x].isna().sum()}, max count: {(sewk.index[0], sewk.iloc[0]) if len(sewk) > 0 else float('nan')}, " + 
                    f"value unique: {df[x].unique()[:5]}"
                )
        self.update_features(columns)
        self.logger.info("END")

    def cut_features_by_correlation(
        self, df: pd.DataFrame=None, cutoff: float=0.99, sample_size: int=10000, dtype: str="float16", is_gpu: bool=False, 
        corr_type: str="pearson", batch_size: int=100, min_n: int=10, **kwargs
    ):
        self.logger.info("START")
        assert df is None or isinstance(df, pd.DataFrame)
        assert cutoff is None or isinstance(cutoff, float) and 0 < cutoff <= 1.0
        if sample_size is None: sample_size = df.shape[0]
        assert isinstance(sample_size, int) and sample_size > 0
        if df is not None:
            df = df.iloc[np.random.permutation(np.arange(df.shape[0]))[:sample_size], :].loc[:, self.columns].copy()
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, dtype: {dtype}, is_gpu: {is_gpu}, corr_type: {corr_type}")
        attr_name = f"features_corr_{corr_type}"
        if df is not None:
            df_corr = get_features_by_correlation(
                df, dtype=dtype, is_gpu=is_gpu, corr_type=corr_type, 
                batch_size=batch_size, min_n=min_n, n_jobs=self.n_jobs, **kwargs
            )
            for i in range(df_corr.shape[0]): df_corr.iloc[i:, i] = float("nan")
            setattr(self, attr_name, df_corr.copy())
        else:
            try: df_corr = getattr(self, attr_name)
            except AttributeError:
                self.logger.warning(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        if cutoff is not None:
            columns_del = df_corr.columns[((df_corr > cutoff) | (df_corr < -cutoff)).sum(axis=0) > 0].values
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del:
                sewk  = df_corr.loc[:, x].copy()
                index = np.where((sewk > cutoff) | (sewk < -cutoff))[0][0]
                self.logger.info(f"feature: {x}, compare: {sewk.index[index]}, corr: {sewk.iloc[index]}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            self.update_features(columns)
        self.logger.info("END")

    def cut_features_by_randomtree_importance(
        self, df: pd.DataFrame=None, cutoff: float=0.9, max_iter: int=1, min_count: int=100, 
        dtype=np.float32, batch_size: int=25, **kwargs
    ):
        self.logger.info("START")
        assert df is None or isinstance(df, pd.DataFrame)
        assert cutoff is None or isinstance(cutoff, float) and 0 < cutoff <= 1.0
        assert len(self.columns_ans.shape) == 1
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, max_iter: {max_iter}, min_count: {min_count}")
        if df is not None:
            df_treeimp = get_features_by_randomtree_importance(
                df, self.columns.tolist(), self.columns_ans[0], dtype=dtype, batch_size=batch_size, 
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
        columns_sort = df_treeimp.index.values.copy()
        self.columns = np.concatenate([columns_sort[isin_compare_string(columns_sort, self.columns)], self.columns[~isin_compare_string(self.columns, columns_sort)]])
        if cutoff is not None:
            columns_del = columns_sort[int(len(columns_sort) * cutoff):]
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del: self.logger.info(f"feature: {x}, ratio: {df_treeimp.loc[x, 'ratio']}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            self.update_features(columns)
        self.logger.info("END")

    def cut_features_by_adversarial_validation(
        self, df_train: pd.DataFrame=None, df_test: pd.DataFrame=None, cutoff: int | float=None, 
        n_split: int=5, n_cv: int=5, dtype=np.float32, batch_size: int=25, **kwargs
    ):
        self.logger.info("START")
        assert df_train is None or isinstance(df_train, pd.DataFrame)
        assert df_test  is None or isinstance(df_test,  pd.DataFrame)
        assert type(df_train) == type(df_test)
        assert cutoff is None or check_type(cutoff, [int, float]) and 0 <= cutoff
        if df_train is not None:
            df_adv, df_pred, se_eval = get_features_by_adversarial_validation(
                df_train, df_test, self.columns.tolist(), columns_ans=None, 
                n_split=n_split, n_cv=n_cv, dtype=dtype, batch_size=batch_size, n_jobs=self.n_jobs, **kwargs
            )
            df_adv = df_adv.sort_values("ratio", ascending=False)
            self.features_adversarial = df_adv.copy()
            self.eval_adversarial_se  = se_eval.copy()
            self.eval_adversarial_df  = df_pred.copy()
        else:
            try: df_adv = getattr(self, "features_adversarial")
            except AttributeError:
                self.logger.warning(f"features_adversarial is not found. Run '{sys._getframe().f_code.co_name}' first.")
                self.logger.info("END")
                return None
        if cutoff is not None:
            columns_del = df_adv.index[df_adv["ratio"] >= cutoff]
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del: self.logger.info(f"feature: {x}, ratio: {df_adv.loc[x, 'ratio']}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            self.update_features(columns)
        self.logger.info("END")

    def cut_features_auto(
        self, df: pd.DataFrame=None, df_test: pd.DataFrame=None, 
        list_proc: list[str] = [
            "self.cut_features_by_variance(df, cutoff=0.99, ignore_nan=False, batch_size=128)",
            "self.cut_features_by_variance(df, cutoff=0.99, ignore_nan=True,  batch_size=128)",
            "self.cut_features_by_randomtree_importance(df, cutoff=None, max_iter=5, min_count=1000, dtype=np.float32, batch_size=25)",
            "self.cut_features_by_adversarial_validation(df, df_test, cutoff=None, n_split=3, n_cv=2, dtype=np.float32, batch_size=25)",
            "self.cut_features_by_correlation(df, cutoff=0.99, dtype='float16', is_gpu=True, corr_type='pearson',  batch_size=2000, min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='spearman', batch_size=500,  min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='kendall',  batch_size=500,  min_n=50, n_sample=250, n_iter=2)",
        ]
    ):
        self.logger.info("START")
        for proc in list_proc:
            eval(proc, {}, {"self": self, "df": df, "df_train": df, "df_test": df_test, "np": np, "pd": pd})
        self.logger.info("END")
    
    def proc_registry(
        self, dict_proc: dict={
            "row": [],
            "exp": [
                '"ProcAsType", np.float32, batch_size=25', 
                '"ProcToValues"', 
                '"ProcReplaceInf", posinf=float("nan"), neginf=float("nan")', 
            ],
            "ans": [
                '"ProcAsType", np.int32',
                '"ProcToValues"',
                '"ProcReshape", (-1, )',
            ]
        }
    ):
        self.logger.info("START")
        assert isinstance(dict_proc, dict)
        for x, y in dict_proc.items():
            assert x in ["row", "exp", "ans"]
            assert check_type_list(y, str)
        self.proc_row = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=False)
        self.proc_exp = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=True)
        self.proc_ans = RegistryProc(n_jobs=self.n_jobs, is_auto_colslct=True)
        for _type in ["row", "exp", "ans"]:
            if _type in dict_proc:
                for x in dict_proc[_type]:
                    eval(f"self.proc_{_type}.register({x})", {}, {"self": self, "np": np, "pd": pd})
        self.proc_check_init()
        self.logger.info("END")
    
    def proc_fit(self, df: pd.DataFrame, is_row: bool=True, is_exp: bool=True, is_ans: bool=True):
        self.logger.info("START")
        assert isinstance(is_row, bool)
        assert isinstance(is_exp, bool)
        assert isinstance(is_ans, bool)
        self.logger.info(f"row: {is_row}. exp: {is_exp}. ans: {is_ans}.")
        self.logger.info("proc fit: row")
        df = self.proc_row.fit(df, check_inout=["class"]) if is_row else df
        self.logger.info(f"df shape: {df.shape}")
        if is_exp == False and is_ans == False: return df
        self.logger.info("proc fit: exp")
        output_x = self.proc_exp.fit(df[self.columns],     check_inout=["row"]) if is_exp else None
        self.logger.info(f"output_x shape: {output_x.shape if output_x is not None else None}")
        self.logger.info("proc fit: ans")
        output_y = self.proc_ans.fit(df[self.columns_ans], check_inout=["row"]) if is_ans else None
        self.logger.info(f"output_y shape: {output_y.shape if output_y is not None else None}")
        self.logger.info("END")
        return output_x, output_y, df.index
    
    def proc_call(self, df: pd.DataFrame, is_row: bool=False, is_exp: bool=True, is_ans: bool=False):
        self.logger.info("START")
        assert isinstance(is_row, bool)
        assert isinstance(is_exp, bool)
        assert isinstance(is_ans, bool)
        self.logger.info(f"row: {is_row}. exp: {is_exp}. ans: {is_ans}.")
        if is_row:
            df = self.proc_row(df)
            self.logger.info(f"df shape: {df.shape}")
        if is_exp == False and is_ans == False: return df
        output_x = self.proc_exp(df) if is_exp else None
        self.logger.info(f"output_x shape: {output_x.shape if output_x is not None else None}")
        output_y = self.proc_ans(df) if is_ans else None
        self.logger.info(f"output_y shape: {output_y.shape if output_y is not None else None}")
        self.logger.info("END")
        return output_x, output_y, df.index

    def proc_check_init(self):
        self.logger.info("START")
        self.proc_row.check_proc(False)
        self.proc_exp.check_proc(False)
        self.proc_exp.is_check = True
        self.proc_ans.check_proc(False)
        self.proc_ans.is_check = True
        self.logger.info("END")
    
    def predict(self, df: pd.DataFrame=None, input_x: np.ndarray=None, is_row: bool=False, is_exp: bool=True, is_ans: bool=False, **kwargs):
        self.logger.info("START")
        self.logger.info(f"kwargs: {kwargs}")
        assert is_exp
        if input_x is None:
            assert isinstance(df, pd.DataFrame)
            input_x, input_y, input_index = self.proc_call(df, is_row=is_row, is_exp=is_exp, is_ans=is_ans)
        else:
            input_y, input_index = None, None
        output = getattr(self.get_model(), self.model_func)(input_x, **kwargs)
        self.logger.info("END")
        return output, input_y, input_index
    
    def set_cvmodel(self, is_calib: bool=False, is_calib_after_cv: bool=False, list_cv: list[int]=None, is_normalize: bool=False, is_binary_fit: bool=False, n_bins: int=None):
        self.logger.info("START")
        assert isinstance(is_calib, bool)
        assert isinstance(is_calib_after_cv, bool)
        assert isinstance(is_normalize, bool)
        assert isinstance(is_binary_fit, bool)
        assert not (is_calib == False and is_calib_after_cv == True)
        assert len(self.list_cv) > 0 and check_type_list(self.list_cv, str)
        assert list_cv is None or (check_type_list(list_cv, int) and sum([str(x) in self.list_cv for x in list_cv]) == len(list_cv))
        assert n_bins is None or isinstance(n_bins, int)
        if not (is_calib == True and is_calib_after_cv == True): assert n_bins is None
        if list_cv is None: list_cv = self.list_cv
        else:               list_cv = np.array(self.list_cv)[list_cv].tolist()
        if is_calib:
            assert self.is_calib == False
            if is_calib_after_cv:
                self.logger.info("calibrate after cv models.")
                self.model_multi = MultiModel([getattr(self, f"model_cv{i}") for i in list_cv], func_predict=self.model_func)
                valid_df = pd.concat([getattr(self, f"eval_valid_df_cv{i}") for i in list_cv], axis=0, ignore_index=True)
                input_x  = valid_df.loc[:, valid_df.columns.str.contains("^predict_proba_", regex=True)].values
                input_y  = valid_df["answer"].values.astype(int)
                if n_bins is None: n_bins=50
                self.calibration(
                    df=None, input_x=input_x, input_y=input_y, model=self.model_multi, model_func=self.model_func,
                    is_use_valid=False, is_predict=False, is_normalize=is_normalize, is_binary_fit=is_binary_fit, n_bins=n_bins
                )
                self.model_multi = self.calibrater
            else:
                self.logger.info("calibrate each cv models.")
                if is_binary_fit: self.logger.warning(f"This parameter is not valid for this mode. is_binary_fit: {is_binary_fit}")
                for i in list_cv:
                    if not hasattr(self, f"model_cv{i}_calib"):
                        logger.raise_error(f"Please run 'calibration_cv_model' first.")
                self.model_multi = MultiModel([getattr(self, f"model_cv{i}_calib") for i in list_cv], func_predict=self.model_func)
        else:
            self.logger.info("cv models without calibration.")
            self.model_multi = MultiModel([getattr(self, f"model_cv{i}") for i in list_cv], func_predict=self.model_func)
        self.is_cvmodel = True
        self.logger.info("END")
    
    def get_model_mode(self):
        if self.is_cvmodel:
            model_mode = "model_multi"
        elif self.is_calib:
            model_mode = "calibrater"
        else:
            model_mode = "model"
        return model_mode

    def get_model(self):
        model_mode = self.get_model_mode()
        self.logger.info(f"model mode: {model_mode}")
        return getattr(self, model_mode)

    def fit(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame=None, is_proc_fit: bool=True, 
        params_fit: str | dict="{}", params_fit_evaldict: dict={}, is_eval_train: bool=False
    ):
        self.logger.info("START")
        assert isinstance(df_train, pd.DataFrame)
        if df_valid is not None: assert isinstance(df_valid, pd.DataFrame)
        for x in self.columns_otr:
            assert np.any(df_train.columns == x)
            if df_valid is not None: assert np.any(df_valid.columns == x)
        assert isinstance(is_proc_fit, bool)
        assert check_type(params_fit, [str, dict])
        assert isinstance(is_eval_train, bool)
        # pre proc
        if is_proc_fit:
            train_x, train_y, train_index = self.proc_fit(df_train, is_row=True, is_exp=True, is_ans=True)
        else:
            train_x, train_y, train_index = self.proc_call(df_train, is_row=True, is_exp=True, is_ans=True)
        valid_x, valid_y = None, None
        if df_valid is not None:
            valid_x, valid_y, valid_index = self.proc_call(df_valid, is_row=True, is_exp=True, is_ans=True)
        if self.model is None: self.logger.raise_error("model is not set.")
        # fit
        if isinstance(params_fit, str):
            assert isinstance(params_fit_evaldict, dict)
            params_fit_evaldict = copy.deepcopy(params_fit_evaldict)
            params_fit_evaldict.update({"_validation_x": valid_x, "_validation_y": valid_y})
            params_fit = eval(params_fit, {}, params_fit_evaldict)
        self.logger.info(f"model: {self.model}, is_reg: {self.is_reg}, fit params: {params_fit}")
        self.logger.info("fitting: start ...")
        self.model.fit(train_x, train_y, **params_fit)
        self.logger.info("fitting: end ...")
        self.is_fit = True
        # eval
        self.logger.info("evaluate model.")
        if valid_x is not None:
            self.logger.info("evaluate valid.")
            se_eval, df_eval = eval_model(valid_x, valid_y, model=self.model, func_predict=self.model_func, is_reg=self.is_reg)
            self.eval_valid_se = se_eval.copy()
            self.eval_valid_df = df_eval.copy()
            self.eval_valid_df["index"] = valid_index
            for x in self.columns_otr:
                self.eval_valid_df[f"otr_{x}"] = df_valid.loc[valid_index, x].copy().values
            for x in se_eval.index:
                self.logger.info(f"{x}: {se_eval.loc[x]}")
        if is_eval_train:
            self.logger.info("evaluate train.")
            se_eval, df_eval = eval_model(train_x, train_y, model=self.model, func_predict=self.model_func, is_reg=self.is_reg)
            self.eval_train_se = se_eval.copy()
            self.eval_train_df = df_eval.copy()
            self.eval_train_df["index"] = train_index
            for x in self.columns_otr:
                self.eval_train_df[f"otr_{x}"] = df_train.loc[train_index, x].copy().values
            for x in se_eval.index:
                self.logger.info(f"{x}: {se_eval.loc[x]}")
        self.logger.info("END")

    def fit_cross_validation(
            self, df_train: pd.DataFrame,
            n_split: int=None, mask_split: np.ndarray=None, cols_multilabel_split: list[str]=None,
            n_cv: int=None, indexes_train: list[np.ndarray]=None, indexes_valid: list[np.ndarray]=None,
            params_fit: str | dict="{}", params_fit_evaldict: dict={},
            is_proc_fit_every_cv: bool=True, is_save_cv_models: bool=False
        ):
        self.logger.info("START")
        assert isinstance(df_train, pd.DataFrame)
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
        is_fit, ndf_oth = False, None
        index_df = df_train.index.copy()
        if cols_multilabel_split is not None:
            ndf_oth = df_train[cols_multilabel_split].copy().values
        df_train = self.proc_fit(df_train, is_row=True, is_exp=False, is_ans=False)
        ndf_bool = index_df.isin(df_train.index)
        if cols_multilabel_split is not None:
            ndf_oth = ndf_oth[ndf_bool]
        if mask_split is not None:
            mask_split = mask_split[ndf_bool]
        if not is_proc_fit_every_cv:
            self.proc_fit(df_train, is_row=True, is_exp=True, is_ans=True)
            is_fit = True
        if n_split is not None:
            assert cols_multilabel_split is None or check_type_list(cols_multilabel_split, str)
            assert (mask_split is None) or (isinstance(mask_split, np.ndarray) and len(mask_split.shape) == 1 and df_train.shape[0] == mask_split.shape[0] and mask_split.dtype in [bool, np.bool_])
            indexes_train, indexes_valid = [], []
            if is_fit: _, ndf_y, _ = self.proc_call(df_train, is_row=False, is_exp=False, is_ans=True)
            else:      _, ndf_y, _ = self.proc_fit( df_train, is_row=False, is_exp=False, is_ans=True)
            indexes     = np.arange(df_train.shape[0], dtype=int)
            if cols_multilabel_split is None:
                self.logger.info(f"Use splitter: StratifiedKFold, n_split: {n_split}")
                splitter = StratifiedKFold(n_splits=n_split)
            else:
                self.logger.info(f"Use splitter: MultilabelStratifiedKFold, n_split: {n_split}, cols_multilabel_split: {cols_multilabel_split}")
                splitter = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
                if len(ndf_y.shape) == 1: ndf_y = ndf_y.reshape(-1, 1)
                ndf_y = np.concatenate([ndf_oth, ndf_y], axis=1)
            if mask_split is not None:
                indexes_mask = indexes[ mask_split].copy()
                indexes      = indexes[~mask_split].copy()
                ndf_y        = ndf_y[~mask_split]
                self.logger.info(f"Use mask split. mask indexes: {indexes_mask}")
            try:
                for i_split, (index_train, index_valid) in enumerate(splitter.split(indexes, ndf_y)):
                    if mask_split is not None: index_train = np.append(index_train, indexes_mask.copy())
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
        for i_cv, (index_train, index_valid) in enumerate(zip(indexes_train, indexes_valid)):
            i_cv += 1
            self.logger.info(f"cross validation : {i_cv} / {n_cv} start...")
            self.reset_model()
            self.fit(
                df_train=df_train.iloc[index_train, :], df_valid=df_train.iloc[index_valid, :],
                is_proc_fit=is_proc_fit_every_cv, params_fit=params_fit, params_fit_evaldict=params_fit_evaldict,
                is_eval_train=False
            )
            self.logger.info(f"cross validation : {i_cv} / {n_cv} end  ...")
            setattr(self, f"eval_valid_df_cv{str(i_cv).zfill(len(str(n_cv)))}", self.eval_valid_df)
            setattr(self, f"eval_valid_se_cv{str(i_cv).zfill(len(str(n_cv)))}", self.eval_valid_se)
            if is_save_cv_models:
                if hasattr(self.model, "copy"):
                    setattr(self, f"model_cv{str(i_cv).zfill(len(str(n_cv)))}", self.model.copy())
                else:
                    setattr(self, f"model_cv{str(i_cv).zfill(len(str(n_cv)))}", copy.deepcopy(self.model))
            if i_cv >= n_cv: break
        self.list_cv = [f"{str(i_cv+1).zfill(len(str(n_cv)))}" for i_cv in range(n_cv)]
        self.logger.info("END")

    def calibration(
        self, df: pd.DataFrame=None, input_x: np.ndarray=None, input_y: np.ndarray=None, model=None, model_func: str=None,
        is_use_valid: bool=False, is_predict: bool=False, is_normalize: bool=False, is_binary_fit: bool=False, n_bins: int=10
    ):
        """
        None::
            If is_predict == False:
                input_x: it must be probabilities for calibration.
                input_y: it must be answer.
            If is_predict == True:
                df: it must be features.
                   or 
                input_x: it must be features.
                input_y: it must be answer.
        """
        self.logger.info("START")
        assert not self.is_reg
        assert self.is_fit
        assert self.is_cvmodel == False
        assert isinstance(is_use_valid,  bool)
        assert isinstance(is_predict,    bool)
        assert isinstance(is_normalize,  bool)
        assert isinstance(is_binary_fit, bool)
        if model      is None: model      = self.model
        if model_func is None: model_func = self.model_func
        if is_use_valid:
            assert is_predict == False
            assert df is None and input_x is None and input_y is None
            assert hasattr(self, "eval_valid_df") and isinstance(self.eval_valid_df, pd.DataFrame) and self.eval_valid_df.shape[0] > 0
            input_x = self.eval_valid_df.loc[:, self.eval_valid_df.columns.str.contains("^predict_proba_", regex=True)].values
            input_y = self.eval_valid_df["answer"].values.astype(int)
        if is_predict:
            if isinstance(df, pd.DataFrame):
                assert input_x is None
                assert input_y is None
                input_x, input_y, _ = self.predict(df=df, input_x=None, is_row=True, is_exp=True, is_ans=True)
            else:
                assert df is None
                assert isinstance(input_x, np.ndarray)
                assert isinstance(input_y, np.ndarray)
                assert input_x.shape[0] == input_y.shape[0]
                input_x, _, _ = self.predict(df=None, input_x=input_x, is_row=True, is_exp=True, is_ans=True)
        else:
            assert df is None
        assert isinstance(input_x, np.ndarray)
        assert isinstance(input_y, np.ndarray)
        assert input_x.shape[0] == input_y.shape[0]
        self.calibrater = Calibrater(model, model_func, is_normalize=is_normalize, is_reg=self.is_reg, is_binary_fit=is_binary_fit)
        self.logger.info("calibration start...")
        self.calibrater.fit(input_x, input_y, n_bins=n_bins)
        self.logger.info("calibration end...")
        self.is_calib = True
        self.logger.info("END")
    
    def calibration_cv_model(self, is_normalize: bool=False, is_binary_fit: bool=False, n_bins: int=10):
        self.logger.info("START")
        assert isinstance(is_normalize,  bool)
        assert isinstance(is_binary_fit, bool)
        assert not self.is_reg
        assert self.is_fit
        assert len(self.list_cv) > 0
        for x in self.list_cv:
            self.logger.info(f"calibration for {x} ...")
            calibrater = Calibrater(getattr(self, f"model_cv{x}"), self.model_func, is_normalize=is_normalize, is_binary_fit=is_binary_fit)
            df         = getattr(self, f"eval_valid_df_cv{x}").copy()
            input_x    = df.loc[:, df.columns.str.contains("^predict_proba_", regex=True)].values
            input_y    = df.loc[:, df.columns == "answer"].values.reshape(-1).astype(int)
            calibrater.fit(input_x, input_y, n_bins=n_bins)
            setattr(self, f"model_cv{x}_calib", calibrater)
        self.logger.info("END")

    def evaluate(self, df_test: pd.DataFrame, columns_ans: str=None, is_store: bool=False, **kwargs):
        self.logger.info("START")
        assert isinstance(is_store, bool)
        if columns_ans is not None: assert isinstance(columns_ans, str)
        test_x, test_y, test_index = self.proc_call(df_test, is_row=True, is_exp=True, is_ans=(True if columns_ans is None else False))
        if columns_ans is not None: test_y = df_test.loc[test_index, columns_ans].values.copy()
        se_eval, df_eval = eval_model(test_x, test_y, model=self.get_model(), is_reg=self.is_reg, func_predict=self.model_func, **kwargs)
        for x in se_eval.index:
            self.logger.info(f"{x}: {se_eval.loc[x]}")
        if is_store:
            self.eval_test_se = se_eval.copy()
            self.eval_test_df = df_eval.copy()
            self.eval_test_df["index"] = test_index
        self.logger.info("END")
        return se_eval, df_eval
    
    def set_n_jobs(self, n_jobs: int):
        self.n_jobs          = n_jobs
        self.proc_row.n_jobs = n_jobs
        self.proc_exp.n_jobs = n_jobs
        self.proc_ans.n_jobs = n_jobs

    def save(self, dirpath: str, filename: str=None, exist_ok: bool=False, remake: bool=False, encoding: str="utf8", is_minimum: bool=False, is_only_log: bool=False):
        self.logger.info("START")
        assert isinstance(dirpath, str)
        assert isinstance(exist_ok, bool)
        assert isinstance(remake, bool)
        assert isinstance(is_minimum, bool)
        assert isinstance(is_only_log, bool)
        dirpath = correct_dirpath(dirpath)
        makedirs(dirpath, exist_ok=exist_ok, remake=remake)
        if filename is None: filename = f"mlmanager_{id(self)}.pickle"
        self.logger.info(f"save file: {dirpath + filename}.")
        if is_minimum:
            if is_only_log == False:
                with open(dirpath + filename + ".min", mode='wb') as f:
                    pickle.dump(self.copy(is_minimum=is_minimum), f, protocol=4)
            with open(dirpath + filename + ".min.log", mode='w', encoding=encoding) as f:
                f.write(self.logger.internal_stream.getvalue())
        else:
            if is_only_log == False:
                with open(dirpath + filename, mode='wb') as f:
                    pickle.dump(self, f, protocol=4)
            with open(dirpath + filename + ".log", mode='w', encoding=encoding) as f:
                f.write(self.logger.internal_stream.getvalue())
        self.logger.info("END")

def load_manager(filepath: str, n_jobs: int) -> MLManager:
    logger.info("START")
    assert isinstance(n_jobs, int)
    logger.info(f"load file: {filepath}")
    with open(filepath, mode='rb') as f:
        manager = pickle.load(f)
    manager.__class__ = MLManager
    manager.logger = set_logger(manager.logger.name, internal_log=True)
    manager.set_n_jobs(n_jobs)
    if os.path.exists(f"{filepath}.log"):
        logger.info(f"load log file: {filepath}.log")
        with open(f"{filepath}.log", mode='r') as f:
            manager.logger.internal_stream.write(f.read())
    manager.logger.info(f"load: {filepath}, jobs: {n_jobs}")
    logger.info("END")
    return manager


class ChainModel:
    def __init__(self, output_string: str, is_normalize: bool=False):
        assert isinstance(output_string, str)
        assert isinstance(is_normalize, bool)
        self.output_string  = output_string
        self.is_normalize   = is_normalize
        self.list_mlmanager: list[MLManager] = []

    def add(self, mlmanager: MLManager, name: str, input_string: str=None):
        """
        Params::
            mlmanager:
                MLManager
            name:
                This name is used to identify which model's answer is.
            input_string:
                This is used to organize input features.
                If use this, you can emmit to be used "prec_call" process.
                You can use same input or add previouos prediction
        Usage::
            add("./test.mlmanager.pickle", "test", "ndf")
        """
        logger.info("START")
        assert isinstance(mlmanager, MLManager)
        assert isinstance(name, str)
        assert input_string is None or isinstance(input_string, str)
        logger.info(f"add name: {name}, input_string: {input_string}")
        self.list_mlmanager.append({"mlmanager": mlmanager, "name": name, "input_string": input_string})
        logger.info("END")

    def predict(self, input_x: np.ndarray | pd.DataFrame, is_row: bool=False, is_exp: bool=True, is_ans: bool=False, is_normalize: bool=None, **kwargs):
        logger.info(f"START {self.__class__}")
        assert len(self.list_mlmanager) > 0
        if isinstance(input_x, pd.DataFrame):
            input_x, _, _ = self.list_mlmanager[0].proc_call(input_x, is_row=is_row, is_exp=is_exp, is_ans=is_ans)
        else:
            assert isinstance(input_x, np.ndarray)
        dict_output = {"ndf": input_x, "np": np, "pd": pd}
        for _, dictwk in enumerate(self.list_mlmanager):
            mlmanager: MLManager = dictwk["mlmanager"]
            model_name           = dictwk["name"]
            input_string         = dictwk["input_string"]
            logger.info(f"model: {model_name} predict.")
            if input_string is None:
                output, _, _ = mlmanager.predict(df=input_x, input_x=None, is_row=is_row, is_exp=is_exp, is_ans=is_ans, **kwargs)
            else:
                input        = eval(input_string, {}, dict_output)
                output, _, _ = mlmanager.predict(df=None, input_x=input, is_row=is_row, is_exp=is_exp, is_ans=is_ans, **kwargs)
            assert model_name not in dict_output
            dict_output[model_name] = output.copy()
        try:
            output = eval(self.output_string, {}, dict_output)
        except Exception as e:
            logger.info(f"{dict_output}")
            logger.raise_error(f"{e}")
        if is_normalize is None: is_normalize = self.is_normalize
        if is_normalize:
            logger.info("normalize output...")
            assert len(output.shape) == 2
            output = output / output.sum(axis=-1).reshape(-1, 1)
        logger.info("END")
        return output
