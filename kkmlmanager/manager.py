import sys, pickle, os, copy
import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.model_selection import StratifiedKFold

# local package
from kkmlmanager.regproc import RegistryProc
from kkmlmanager.features import get_features_by_variance, get_features_by_correlation, get_features_by_randomtree_importance, get_features_by_adversarial_validation
from kkmlmanager.eval import eval_model
from kkmlmanager.calibration import Calibrater, calibration_curve_plot
from kkmlmanager.util.numpy import isin_compare_string
from kkmlmanager.util.com import check_type, check_type_list, correct_dirpath, makedirs
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "MLManager",
    "load_manager",
    "MultiModel",
]


class MLManager:
    def __init__(
        self,
        # model parameter
        columns_exp: List[str], columns_ans: Union[str, List[str]], columns_otr: List[str]=None, is_reg: bool=False, 
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
        self.model_multi  = None
        self.is_cvmodel   = False
        self.calibrater   = None
        self.is_fit       = False
        self.is_calib     = False
        self.list_cv      = []
        self.columns_hist = [self.columns_exp.copy(), ]
        self.columns      = self.columns_exp.copy()
        self.proc_row     = RegistryProc(n_jobs=self.n_jobs)
        self.proc_exp     = RegistryProc(n_jobs=self.n_jobs)
        self.proc_ans     = RegistryProc(n_jobs=self.n_jobs)
        self.proc_check_init()
        self.logger.info("END")
    
    def copy(self, is_minimum: bool=False):
        assert isinstance(is_minimum, bool)
        if is_minimum:
            ins = __class__(
                self.columns_exp.tolist(), self.columns_ans.tolist(), columns_otr=self.columns_otr.tolist(), 
                is_reg=self.is_reg, outdir=self.outdir, random_seed=self.random_seed, n_jobs=self.n_jobs
            )
            ins.model        = copy.deepcopy(self.model)
            ins.model_class  = copy.deepcopy(self.model_class)
            ins.model_args   = copy.deepcopy(self.model_args)
            ins.model_kwargs = copy.deepcopy(self.model_kwargs)
            ins.model_func   = copy.deepcopy(self.model_func)
            ins.model_multi  = copy.deepcopy(self.model_multi)
            ins.is_cvmodel   = self.is_cvmodel
            if ins.is_cvmodel: ins.model       = None
            else:              ins.model_multi = None
            ins.calibrater   = copy.deepcopy(self.calibrater)
            ins.is_fit       = self.is_fit
            ins.is_calib     = self.is_calib
            if ins.is_calib:
                ins.model       = None
                ins.model_multi = None
            ins.list_cv      = copy.deepcopy(self.list_cv)
            ins.columns_hist = copy.deepcopy(self.columns_hist)
            ins.columns      = self.columns.copy()
            ins.proc_row     = copy.deepcopy(self.proc_row)
            ins.proc_exp     = copy.deepcopy(self.proc_exp)
            ins.proc_ans     = copy.deepcopy(self.proc_ans)
            return ins
        else:
            return copy.deepcopy(self)

    def set_model(self, model, *args, model_func_predict: str=None, **kwargs):
        self.logger.info("START")
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

    def update_features(self, features: Union[List[str], np.ndarray]):
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
                logger.raise_error(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.")
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
                logger.raise_error(f"{attr_name} is not found. Run '{sys._getframe().f_code.co_name}' first.")
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
                logger.raise_error(f"features_treeimp is not found. Run '{sys._getframe().f_code.co_name}' first.")
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
        self, df_train: pd.DataFrame=None, df_test: pd.DataFrame=None, cutoff: Union[int, float]=None, 
        n_split: int=5, n_cv: int=5, dtype=np.float32, batch_size: int=25, **kwargs
    ):
        self.logger.info("START")
        assert df_train is None or isinstance(df_train, pd.DataFrame)
        assert df_test  is None or isinstance(df_test,  pd.DataFrame)
        assert type(df_train) == type(df_test)
        assert cutoff is None or check_type(cutoff, [int, float]) and 0 <= cutoff
        if df_train is not None:
            df_adv, _ = get_features_by_adversarial_validation(
                df_train, df_test, self.columns.tolist(), columns_ans=None, 
                n_split=n_split, n_cv=n_cv, dtype=dtype, batch_size=batch_size, n_jobs=self.n_jobs, **kwargs
            )
            df_adv = df_adv.sort_values("ratio", ascending=False)
            self.features_adversarial = df_adv.copy()
        else:
            try: df_adv = getattr(self, "features_adversarial")
            except AttributeError:
                logger.raise_error(f"features_adversarial is not found. Run '{sys._getframe().f_code.co_name}' first.")
        if cutoff is not None:
            columns_del = df_adv.index[df_adv["ratio"] >= cutoff]
            columns_del = self.columns[isin_compare_string(self.columns, columns_del)]
            for x in columns_del: self.logger.info(f"feature: {x}, ratio: {df_adv.loc[x, 'ratio']}")
            columns = self.columns[~isin_compare_string(self.columns, columns_del)]
            self.update_features(columns)
        self.logger.info("END")

    def cut_features_auto(
        self, df: pd.DataFrame=None, df_test: pd.DataFrame=None, 
        list_proc: List[str] = [
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
        self.proc_row = RegistryProc(n_jobs=self.n_jobs)
        self.proc_exp = RegistryProc(n_jobs=self.n_jobs)
        self.proc_ans = RegistryProc(n_jobs=self.n_jobs)
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
        df = df[self.columns.tolist() + self.columns_ans.tolist() + self.columns_otr[~np.isin(self.columns_otr, self.columns_ans)].tolist()].copy()
        df = self.proc_row.fit(df, check_inout=["class"]) if is_row else df
        if is_exp == False and is_ans == False: return df
        output_x = self.proc_exp.fit(df[self.columns],     check_inout=["row"]) if is_exp else None
        output_y = self.proc_ans.fit(df[self.columns_ans], check_inout=["row"]) if is_ans else None
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
        if is_exp == False and is_ans == False: return df
        output_x = self.proc_exp(df) if is_exp else None
        output_y = self.proc_ans(df) if is_ans else None
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
    
    def predict(self, df: pd.DataFrame, is_row: bool=False, is_exp: bool=True, is_ans: bool=False, **kwargs):
        self.logger.info("START")
        assert is_exp
        input_x, input_y, input_index = self.proc_call(df, is_row=is_row, is_exp=is_exp, is_ans=is_ans)
        self.logger.info(f"predict mode: {'calib' if self.is_calib else 'normal'}")
        output = getattr(self.get_model(), self.model_func)(input_x, **kwargs)
        self.logger.info("END")
        return output, input_y, input_index
    
    def set_cvmodel(self):
        self.logger.info("START")
        assert len(self.list_cv) > 0
        self.model_multi = MultiModel([getattr(self, f"model_cv{i}") for i in self.list_cv], func_predict=self.model_func)
        self.is_cvmodel = True
        self.logger.info("END")
    
    def get_model(self, calib: bool=True):
        if self.is_cvmodel:
            model_mode = "model_multi"
        else:
            model_mode = "model"
        if self.is_calib and calib:
            model_mode = "calibrater"
        return getattr(self, model_mode)

    def fit(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame=None, is_proc_fit: bool=True, 
        params_fit: Union[str, dict]="{}", params_fit_evaldict: dict={}, is_eval_train: bool=False
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
            self, df_train: pd.DataFrame, n_split: int=None, n_cv: int=None,
            indexes_train: List[np.ndarray]=None, indexes_valid: List[np.ndarray]=None,
            params_fit: Union[str, dict]="{}", params_fit_evaldict: dict={},
            is_proc_fit_every_cv: bool=True, is_save_model: bool=False
        ):
        self.logger.info("START")
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(is_proc_fit_every_cv, bool)
        assert isinstance(is_save_model, bool)
        if n_split is None:
            assert indexes_train is not None and check_type_list(indexes_train, np.ndarray)
            assert indexes_valid is not None and check_type_list(indexes_valid, np.ndarray)
            assert len(indexes_train) == len(indexes_valid)
            n_cv = len(indexes_train) if n_cv is None else n_cv
        else:
            assert isinstance(n_split, int) and n_split >= 2
            assert isinstance(n_cv,    int) and n_cv    >= 1 and n_cv <= n_split
        df_train = self.proc_fit(df_train, is_row=True, is_exp=False, is_ans=False)
        if not is_proc_fit_every_cv:
            self.proc_fit(df_train, is_row=False, is_exp=True, is_ans=True)
        if n_split is not None:
            indexes_train, indexes_valid = [], []
            splitter = StratifiedKFold(n_splits=n_split)
            _, ndf_y, _ = self.proc_fit(df_train, is_row=False, is_exp=False, is_ans=True)
            try:
                for index_train, index_test in splitter.split(np.arange(df_train.shape[0], dtype=int), ndf_y):
                    indexes_train.append(index_train)
                    indexes_valid.append(index_test )
            except ValueError as e:
                logger.warning(f"{e}")
                logger.info("use normal random splitter.")
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
            if is_save_model:
                setattr(self, f"model_cv{str(i_cv).zfill(len(str(n_cv)))}", copy.deepcopy(self.model))
            if i_cv >= n_cv: break
        self.list_cv = [f"{str(i_cv+1).zfill(len(str(n_cv)))}" for i_cv in range(n_cv)]
        self.logger.info("END")

    def calibration(self, df_calib: pd.DataFrame=None, columns_ans: str=None, n_bins: int=10, is_fit_by_class: bool=True, is_cvmode: bool=False):
        self.logger.info("START")
        assert not self.is_reg
        assert self.is_fit
        if df_calib is None:
            assert len(self.list_cv) > 0
        else:
            assert isinstance(df_calib, pd.DataFrame)
        if is_cvmode:
            assert len(self.list_cv) > 0
            assert df_calib is None
            calibrater = MultiModel([Calibrater(getattr(self, f"model_cv{i}"), self.model_func, is_fit_by_class=is_fit_by_class) for i in self.list_cv])
            for i, i_cv in enumerate(self.list_cv):
                df = getattr(self, f"eval_valid_df_cv{i_cv}").copy()
                input_x = df.loc[:, df.columns.str.contains("^predict_proba", regex=True)].values
                if columns_ans is None: columns_ans = "answer"
                assert isinstance(columns_ans, str) and np.any(df.columns == columns_ans)
                input_y = df.loc[:, df.columns == columns_ans].values.reshape(-1).astype(int)
                calibrater.models[i].fit(input_x, input_y)
        else:
            calibrater = Calibrater(self.get_model(calib=False), self.model_func, is_fit_by_class=is_fit_by_class)
            # fitting
            input_x, input_y = None, None
            if df_calib is None:
                df      = pd.concat([getattr(self, f"eval_valid_df_cv{x}") for x in self.list_cv], axis=0, ignore_index=True, sort=False)
                input_x = df.loc[:, df.columns.str.contains("^predict_proba", regex=True)].values
                if columns_ans is None: columns_ans = "answer"
                assert isinstance(columns_ans, str) and np.any(df.columns == columns_ans)
                input_y = df.loc[:, df.columns == columns_ans].values.reshape(-1).astype(int)
            else:
                input_x, input_y, _ = self.predict(df_calib, is_row=False, is_exp=True, is_ans=True)
            self.logger.info("calibration start...")
            calibrater.fit(input_x, input_y)
            self.logger.info("calibration end...")
        self.calibrater = calibrater
        output          = getattr(self.calibrater, self.model_func)(input_x, is_mock=True)
        self.is_calib   = True
        self.calib_fig  = calibration_curve_plot(input_x, output, input_y, n_bins=n_bins)
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
        return df_eval, se_eval

    def save(self, dirpath: str, filename: str=None, exist_ok: bool=False, remake: bool=False, encoding: str="utf8", is_minimum: bool=False):
        self.logger.info("START")
        assert isinstance(dirpath, str)
        assert isinstance(exist_ok, bool)
        assert isinstance(remake, bool)
        assert isinstance(is_minimum, bool)
        dirpath = correct_dirpath(dirpath)
        makedirs(dirpath, exist_ok=exist_ok, remake=remake)
        if filename is None: filename = f"mlmanager_{id(self)}.pickle"
        self.logger.info(f"save file: {dirpath + filename}.")
        if is_minimum:
            with open(dirpath + filename + ".min", mode='wb') as f:
                pickle.dump(self.copy(is_minimum=is_minimum), f, protocol=4)
            with open(dirpath + filename + ".min.log", mode='w', encoding=encoding) as f:
                f.write(self.logger.internal_stream.getvalue())
        else:
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
    manager.n_jobs = n_jobs
    manager.proc_row.n_jobs = n_jobs
    manager.proc_exp.n_jobs = n_jobs
    manager.proc_ans.n_jobs = n_jobs
    if os.path.exists(f"{filepath}.log"):
        logger.info(f"load log file: {filepath}.log")
        with open(f"{filepath}.log", mode='r') as f:
            manager.logger.internal_stream.write(f.read())
    logger.info("END")
    return manager


class MultiModel:
    def __init__(self, models: List[object], func_predict: str=None):
        assert isinstance(models, list) and len(models) > 0
        self.classes_ = models[0].classes_ if hasattr(models[0], "classes_") else None
        for model in models:
            if func_predict is None:
                assert hasattr(model, "predict")
                assert hasattr(model, "predict_proba")
            else:
                assert hasattr(model, func_predict)
            if self.classes_ is not None:
                assert np.all(self.classes_ == model.classes_)
        self.models = models
        if func_predict is not None:
            setattr(
                self, func_predict, 
                lambda input, weight=None, **kwargs: self.predict_common(input, weight=weight, funcname=func_predict, **kwargs)
            )
        self.func_predict = func_predict
    def predict_common(self, input: np.ndarray, weight: List[float]=None, funcname: str="predict", **kwargs):
        logger.info("START")
        logger.info(f"func predict name: {funcname}")
        assert isinstance(input, np.ndarray)
        if weight is None: weight = [1.0] * len(self.models)
        assert check_type_list(weight, float)
        assert isinstance(funcname, str)
        output = None
        for i, model in enumerate(self.models):
            logger.info(f"predict model {i}")
            _output = getattr(model, funcname)(input, **kwargs) * weight[i]
            if output is None: output  = _output
            else:              output += _output
        output = output / sum(weight)
        logger.info("END")
        return output
    def predict(self, input: np.ndarray, weight: List[float]=None, **kwargs):
        return self.predict_common(input, weight=weight, funcname="predict", **kwargs)
    def predict_proba(self, input: np.ndarray, weight: List[float]=None, **kwargs):
        return self.predict_common(input, weight=weight, funcname="predict_proba", **kwargs)
