import sys, pickle, os
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Union
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

# local package
from kkmlmanager.features import get_features_by_variance, get_features_by_correlation, get_features_by_randomtree_importance, get_features_by_adversarial_validation
from kkmlmanager.util.numpy import isin_compare_string
from kkmlmanager.util.com import check_type, check_type_list, correct_dirpath, makedirs
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "MLManager",
    "load_manager"
]


class MLManager:
    def __init__(
        self,
        # model parameter
        columns_exp: List[str], columns_ans: Union[str, List[str]], columns_oth: List[str]=None,
        # common parameter
        outdir: str="./output/", random_seed: int=1, n_jobs: int=1
    ):
        self.logger = set_logger(f"{__class__.__name__}.{id(self)}", internal_log=True)
        self.logger.info("START")
        if isinstance(columns_ans, str): columns_ans = [columns_ans, ]
        if columns_oth is None: columns_oth = []
        assert check_type_list(columns_exp, str)
        assert check_type_list(columns_ans, str)
        assert check_type_list(columns_oth, str)
        assert isinstance(outdir, str)
        assert isinstance(random_seed, int) and random_seed >= 0
        assert isinstance(n_jobs, int) and n_jobs >= 1
        self.columns_exp = np.array(columns_exp)
        self.columns_ans = np.array(columns_ans)
        self.columns_oth = np.array(columns_oth)
        self.outdir      = correct_dirpath(outdir)
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self.initialize()
        self.logger.info("END")

    def __str__(self):
        return f"model: {self.model}\ncolumns explain: {self.columns_exp}\ncolumns answer: {self.columns_ans}\ncolumns other: {self.columns_oth}"
    
    def initialize(self):
        self.logger.info("START")
        self.model        = None
        self.model_args   = None
        self.model_kwargs = None
        self.is_fit       = False
        self.columns_hist = [self.columns_exp.copy(), ]
        self.columns      = self.columns_exp.copy()
        self.logger.info("END")

    def set_model(self, model, *args, **kwargs):
        self.logger.info("START")
        self.initialize()
        self.model        = model(*args, **kwargs)
        self.model_args   = args
        self.model_kwargs = kwargs
        self.is_fit       = False
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
        assert isinstance(sample_size, int) and sample_size > 0
        df = df.iloc[np.random.permutation(np.arange(df.shape[0]))[:sample_size], :].copy()
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, dtype: {dtype}, is_gpu: {is_gpu}, corr_type: {corr_type}")
        attr_name = f"features_corr_{corr_type}"
        if df is not None:
            df_corr = get_features_by_correlation(
                df[self.columns], dtype=dtype, is_gpu=is_gpu, corr_type=corr_type, 
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
        self, df: pd.DataFrame=None, cutoff: float=0.9, max_iter: int=1, min_count: int=100, **kwargs
    ):
        self.logger.info("START")
        assert df is None or isinstance(df, pd.DataFrame)
        assert cutoff is None or isinstance(cutoff, float) and 0 < cutoff <= 1.0
        assert len(self.columns_ans.shape) == 1
        self.logger.info(f"df: {df.shape if df is not None else None}, cutoff: {cutoff}, max_iter: {max_iter}, min_count: {min_count}")
        if df is not None:
            is_reg = False if df[self.columns_ans].dtypes[0] in [int, np.int16, np.int32, np.int64] else True
            df_treeimp = get_features_by_randomtree_importance(
                df, self.columns.tolist(), self.columns_ans[0], is_reg=is_reg, max_iter=max_iter, 
                min_count=min_count, n_jobs=self.n_jobs, **kwargs
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
        self, df_train: pd.DataFrame=None, df_test: pd.DataFrame=None, cutoff: Union[int, float]=None, n_split: int=5, n_cv: int=5, **kwargs
    ):
        self.logger.info("START")
        assert df_train is None or isinstance(df_train, pd.DataFrame)
        assert df_test  is None or isinstance(df_test,  pd.DataFrame)
        assert type(df_train) == type(df_test)
        assert cutoff is None or check_type(cutoff, [int, float]) and 0 <= cutoff
        if df_train is not None:
            df_adv, _ = get_features_by_adversarial_validation(
                df_train, df_test, self.columns.tolist(), columns_ans=None, 
                n_split=n_split, n_cv=n_cv, n_jobs=self.n_jobs, **kwargs
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
            "self.cut_features_by_randomtree_importance(df, cutoff=None, max_iter=5, min_count=1000)",
            "self.cut_features_by_adversarial_validation(df, df_test, cutoff=None, n_split=3, n_cv=2)",
            "self.cut_features_by_correlation(df, cutoff=0.99, dtype='float16', is_gpu=True, corr_type='pearson',  batch_size=2000, min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='spearman', batch_size=500,  min_n=100)",
            "self.cut_features_by_correlation(df, cutoff=None, dtype='float16', is_gpu=True, corr_type='kendall',  batch_size=500,  min_n=50, n_sample=250, n_iter=2)",
        ]
    ):
        for proc in list_proc:
            eval(proc, {}, {"self": self, "df": df, "df_train": df, "df_test": df_test})

    def eval_model(self, df_score, **eval_params) -> (pd.DataFrame, pd.Series, ):
        """
        回帰や分類を意識せずに評価値を出力する
        """
        df_conf, se_eval = pd.DataFrame(), pd.Series(dtype=object)
        if self.is_classification_model():
            df_conf, se_eval = eval_classification_model(df_score, "answer", "predict", ["predict_proba_"+str(int(_x)) for _x in self.model.classes_], labels=self.model.classes_, **eval_params)
        else:
            df_conf, se_eval = eval_regressor_model(     df_score, "answer", "predict", **eval_params)
        return df_conf, se_eval


    def fit(
            self, df_train: pd.DataFrame, df_valid: pd.DataFrame=None, 
            fit_params: dict={}, eval_params: dict={"n_round":3, "eval_auc_dict":{}}, 
            pred_params: dict={"do_estimators":False}, is_preproc_fit: bool=True, 
            is_pred_train: bool=True, is_pred_valid: bool=True
        ):
        """
        モデルのfit関数. fitだけでなく、評価値や重要度も格納する.
        Params::
            df_train: 訓練データ
            df_valid: 検証データ
            fit_params:
                fit 時の追加のparameter を渡す事ができる. 特殊文字列として_validation_x, _validation_y があり、
                これは交差検証時のvalidation dataをその文字列と変換して渡す事ができる
                {"eval_set":[("_validation_x", "_validation_y")], "early_stopping_rounds":50}
            eval_params:
                評価時のparameter. {"n_round":3, "eval_auc_list":{}}
                eval_classification_model など参照
            pred_params:
                predict_detail など参照
            is_preproc_fit: preproc.fit を事前に df_train で行うかどうか
            is_pred_train: train の predict を保存するかどうか（サイズが大きくなるので）
            is_pred_valid: valid の predict を保存するかどうか（サイズが大きくなるので）
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df_train.shape}, fit_params:{fit_params}, eval_params:{eval_params}")
        self.logger.info(f"features:{self.colname_explain.shape}")
        if self.model is None: self.logger.raise_error("model is not set.")

        fit_params  = fit_params. copy() if fit_params  is not None else {}
        eval_params = eval_params.copy() if eval_params is not None else {}
        pred_params = pred_params.copy() if pred_params is not None else {}

        # 前処理
        if is_preproc_fit: self.preproc.fit(df_train)
        X_train, Y_train, index_train_df = self.preproc(df_train, autofix=True, ret_index=True)
        X_valid, Y_valid, index_valid_df = None, None, None
        if df_valid is not None: X_valid, Y_valid, index_valid_df = self.preproc(df_valid, autofix=True, ret_index=True)
        fit_params = conv_validdata_in_fitparmas(fit_params.copy(), X_valid, Y_valid) # validation data set.
        print(fit_params)
        # fit
        self.logger.debug("create model by All samples : start ...")
        self.model.fit(X_train, Y_train, **fit_params)
        self.logger.debug("create model by All samples : end ...")
        # 情報の保存
        if len(Y_train.shape) == 1 and Y_train.dtype in [np.int8, np.int16, np.int32, np.int64]:
            sewk = pd.DataFrame(Y_train).groupby(0).size().sort_index()
            if is_callable(self.model, "classes_") == False:
                ## classes_ がない場合は手動で追加する
                self.model.classes_ = np.sort(np.unique(Y_train)).astype(int)
            self.n_trained_samples = {int(x):sewk[int(x)] for x in self.model.classes_}

        # 精度の記録
        pred_params["n_jobs"] = self.n_jobs
        ## 訓練データ
        df_score = predict_detail(self.model, X_train, y=Y_train, **pred_params)
        df_score["index"]  = index_train_df.copy()
        for x in self.colname_other: df_score["other_"+x] = df_train.loc[:, x].copy().values
        self.index_train   = index_train_df.copy() # DataFrame のインデックスを残しておく
        if is_pred_train: self.df_pred_train = df_score.copy()
        if df_score.columns.isin(["predict", "answer"]).sum() == 2:
            df_conf, se_eval = self.eval_model(df_score, **eval_params)
            self.logger.info("evaluation model by train data.")
            self.df_cm_train   = df_conf.copy()
            self.se_eval_train = se_eval.astype(str).copy()
            self.logger.info(f'\n{self.df_cm_train}\n{self.se_eval_train}')
        ## 検証データ
        if df_valid is not None:
            df_score = predict_detail(self.model, X_valid, y=Y_valid, **pred_params)
            df_score["index"]  = index_valid_df.copy()
            for x in self.colname_other: df_score["other_"+x] = df_valid.loc[:, x].copy().values
            self.index_valid = index_valid_df.copy()
            if is_pred_valid: self.df_pred_valid = df_score.copy()
            if df_score.columns.isin(["predict", "answer"]).sum() == 2:
                df_conf, se_eval = self.eval_model(df_score, **eval_params)
                self.logger.info("evaluation model by train data.")
                self.df_cm_valid  = df_conf.copy()
                self.se_eval_valid = se_eval.astype(str).copy()
                self.logger.info(f'\n{self.df_cm_valid}\n{self.se_eval_valid}')
        
        # 特徴量の重要度
        self.logger.info("feature importance saving...")
        if is_callable(self.model, "feature_importances_"):
            if len(self.colname_explain) == len(self.model.feature_importances_):
                _df = pd.DataFrame(np.array([self.colname_explain, self.model.feature_importances_]).T, columns=["feature_name","importance"])
                _df = _df.sort_values(by="importance", ascending=False).reset_index(drop=True)
                self.df_feature_importances = _df.copy()
            else:
                self.logger.warning("feature importance not saving...")
        self.is_model_fit = True
        self.logger.info("END")


    def fit_cross_validation(
            self, df_learn: pd.DataFrame, indexes_train: List[np.ndarray], indexes_valid: List[np.ndarray], 
            indexes_select: List[int]=None,
            fit_params={}, eval_params={"n_round":3, "eval_auc_dict":{}}, pred_params={"do_estimators":False}, 
            is_save_traindata: bool=False, get_model_attribute: str= None, is_preproc_fit: bool=True, is_save_valid: bool=True
        ):
        """
        交差検証を行い、モデルを作成する.
        ※ほとんどは fit 関数と共通するのでそちらを参照する
        Params::
            df_learn: 訓練データ
            indexes_train: list 形式の df.loc[xxx] でアクセスできる index が格納されたデータ
            indexes_valid: list 形式の df.loc[xxx] でアクセスできる index が格納されたデータ
            is_save_traindata: 訓練データを格納するかどうか
            get_model_attribute: 交差検証中に fit後の model から取りたいデータがあればここで attribute を指定する
        """
        self.logger.info("START")
        self.logger.info(
            f'df shape: {df_learn.shape}, indexes_train: {indexes_train}, indexes_valid: {indexes_valid}, ' +
            f'fit_params: {fit_params}, fit_params_treval_params: {eval_params}, pred_params: {pred_params}, ' + 
            f'is_save_traindata: {is_save_traindata}, get_model_attribute: {get_model_attribute}'
        )
        self.logger.info(f"features: {self.colname_explain.shape}")
        if self.model is None:
            self.logger.raise_error("model is not set.")

        # 交差検証開始
        df_score_train, df_score_valid, list_model_attribute = [], [], []
        for i_valid, (index_train, index_valid) in enumerate(zip(indexes_train, indexes_valid)):
            self.logger.info(f"Cross Validation : {i_valid} start...")
            if indexes_select is None or i_valid in indexes_select:
                df_train = df_learn.loc[index_train, :].copy()
                df_valid = df_learn.loc[index_valid, :].copy()
                self.fit(df_train, df_valid=df_valid, fit_params=fit_params, eval_params=eval_params, pred_params=pred_params, is_preproc_fit=is_preproc_fit, is_pred_train=is_save_traindata)
                self.df_pred_train["i_valid"] = i_valid
                self.df_pred_valid["i_valid"] = i_valid
                if is_save_valid: self.save(name=(self.name + f".{i_valid}"), mode=1, exist_ok=True, remake=False)
                if is_save_traindata: df_score_train.append(self.df_pred_train.copy())
                df_score_valid.append(self.df_pred_valid.copy())
                ## validation の model の中で取りたい変数があればここで取得する
                if get_model_attribute is not None:
                    list_model_attribute.append(self.model.__getattribute__(get_model_attribute))
                self.logger.info(f"Cross Validation : {i_valid} end...")
            else:
                self.logger.info(f"Cross Validation : {i_valid} skip...")
        if is_save_traindata:
            df_score_train     = pd.concat(df_score_train, axis=0, sort=False, ignore_index=True)
            self.df_pred_train = df_score_train.copy()
            self.index_train   = self.df_pred_train["index"].values # DaaFrame のインデックスを残しておく
        df_score_valid        = pd.concat(df_score_valid, axis=0, sort=False, ignore_index=True)
        self.df_pred_valid_cv = df_score_valid.copy()
        self.index_valid_cv   = self.df_pred_valid_cv["index"].values # DaaFrame のインデックスを残しておく
        # 検証データを評価する(交差検証では訓練データの記録は残しにくいので残さない)
        if self.df_pred_valid_cv.columns.isin(["predict", "answer"]).sum() == 2:
            df_conf, se_eval = self.eval_model(self.df_pred_valid_cv, **eval_params)
            self.logger.info("evaluation model by train data.")
            self.df_cm_valid_cv   = df_conf.copy()
            self.se_eval_valid_cv = se_eval.astype(str).copy()
            self.logger.info(f'\n{self.df_cm_valid_cv}\n{self.se_eval_valid_cv}')
        self.logger.info("END")


    def calibration(self):
        """
        予測確率のキャリブレーション
        予測モデルは作成済みで別データでfittingする場合
        """
        self.logger.info("START")
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")
        if self.is_classification_model() == False:
            self.logger.raise_error("model type is not classification")
        self.calibrater = Calibrater(self.model)
        self.logger.info("\n%s", self.calibrater)
        ## fittingを行う
        df = self.df_pred_valid_cv.copy()
        if df.shape[0] == 0:
            ## クロスバリデーションを行っていない場合はエラー
            self.logger.raise_error("cross validation is not done !!")
        X = df.loc[:, df.columns.str.contains("^predict_proba_")].values
        Y = df["answer"].values
        self.calibrater.fit(X, Y)
        self.is_calibration = True # このフラグでON/OFFする
        # ここでのXは確率なので、mockを使って補正後の値を確認する
        pred_prob_bef = X
        pred_prob_aft = self.calibrater.predict_proba_mock(X)
        self.is_calibration = True # このフラグでON/OFFする
        # Calibration Curve Plot
        classes = np.sort(np.unique(Y).astype(int))
        self.fig["calibration_curve"] = plt.figure(figsize=(12, 8))
        ax1 = self.fig["calibration_curve"].add_subplot(2,1,1)
        ax2 = self.fig["calibration_curve"].add_subplot(2,1,2)
        ## ラベル数に応じて処理を分ける
        for i, i_class in enumerate(classes):
            ## 変化の前後を記述する
            fraction_of_positives, mean_predicted_value = calibration_curve((Y==i_class), pred_prob_bef[:, i], n_bins=10)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s:", label="before_label_"+str(i_class))
            ax2.hist(pred_prob_bef[:, i], range=(0, 1), bins=10, label="before_label_"+str(i_class), histtype="step", lw=2)
            fraction_of_positives, mean_predicted_value = calibration_curve((Y==i_class), pred_prob_aft[:, i], n_bins=10)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="after_label_"+str(i_class))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.legend()
        ax2.legend()
        ax1.set_title('Calibration plots  (reliability curve)')
        ax2.set_xlabel('Mean predicted value')
        ax1.set_ylabel('Fraction of positives')
        ax2.set_ylabel('Count')

        self.logger.info("END")


    def predict(self, df: pd.DataFrame=None, _X: np.ndarray=None, _Y: np.ndarray=None, pred_params={"do_estimators":False}, row_proc: bool=True):
        """
        モデルの予測
        Params::
            df: input DataFrame
            _X: ndarray (こちらが入力されたらpre_proc_などの処理が短縮される)
            _Y: ndarray (こちらが入力されたらpre_proc_などの処理が短縮される)
        """
        self.logger.info("START")
        self.logger.info("df:%s, X:%s, Y:%s, pred_params:%s", \
                         df if df is None else df.shape, \
                         _X if _X is None else _X.shape, \
                         _Y if _Y is None else _Y.shape, \
                         pred_params)
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")

        # 下記の引数パターンの場合はエラーとする
        if (type(df) == type(None)) and (type(_X) == type(None)):
            self.logger.raise_error("df and _X is null.")

        pred_params  = pred_params.copy()
        pred_params["n_jobs"] = self.n_jobs
            
        # 前処理の結果を反映する
        X = None
        if df is not None and row_proc: df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True) # ここで変換しないとdf_scoreとのindexが合わない
        if _X is None:
            X, _ = self.preproc(df, x_proc=True, y_proc=False, row_proc=False, autofix=True)
        else:
            # numpy変換処理での遅延を避けるため、引数でも指定できるようにする
            X = _X

        # 予測処理
        ## キャリブレーターの有無で渡すモデルを変える
        df_score = None
        if self.is_calibration:
            self.logger.info("predict with calibrater.")
            df_score = predict_detail(self.calibrater, X, **pred_params)
        else:
            self.logger.info("predict with no calibrater.")
            df_score = predict_detail(self.model,      X, **pred_params)

        if df is not None:
            df_score["index"] = df.index.values.copy()
            for x in self.colname_other: df_score["other_"+x] = df[x].copy().values

        try:
            Y = None
            if _Y is None and df is not None:
                _, Y = self.preproc(df, x_proc=False, y_proc=True, row_proc=False, autofix=True)
            else:
                # numpy変換処理での遅延を避けるため、引数でも指定できるようにする
                Y = _Y
            if   len(Y.shape) == 1: df_score["answer"] = Y
            elif len(Y.shape) == 2: df_score[[f"answer{i}" for i in range(Y.shape[1])]] = Y
        except KeyError:
            self.logger.warning("answer is none. predict only.")
        except ValueError:
            # いずれは削除する
            self.logger.warning("Preprocessing answer's label is not work.")

        self.logger.info("END")
        return df_score


    # テストデータでの検証. 結果は上位に返却
    def predict_testdata(
            self, df_test, store_eval=False, eval_params={"n_round":3, "eval_auc_list":[]},
            pred_params={"do_estimators":False}, row_proc: bool=True
        ):
        """
        テストデータを予測し、評価する
        """
        self.logger.info("START")
        self.logger.info("df:%s, store_eval:%s, eval_params:%s, pred_params:%s", \
                         df_test.shape, store_eval, eval_params, pred_params)
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")
        eval_params  = eval_params.copy()
        pred_params  = pred_params.copy()
        # 予測する
        df_score = self.predict(df=df_test, pred_params=pred_params, row_proc=row_proc)
        df_score["i_split"] = 0
        for x in self.colname_other: df_score["other_"+x] = df_test[x].values
        self.df_pred_test = df_score.copy()
        ## 評価値を格納
        df_conf, se_eval = self.eval_model(df_score, **eval_params)
        self.logger.info("\n%s", df_conf)
        self.logger.info("\n%s", se_eval)
        colname_answer, _y_test = df_score.columns[df_score.columns.str.contains("^answer", regex=True)].tolist(), None
        if len(colname_answer) > 0:
            _y_test = df_score[colname_answer].values
            if store_eval == True:
                ## データを格納する場合(下手に上書きさせないためデフォルトはFalse)
                self.df_cm_test   = df_conf.copy()
                self.se_eval_test = se_eval.copy()
                ## テストデータの割合を格納
                if len(_y_test.shape) == 2 and _y_test.shape[1] == 1: _y_test = _y_test.reshape(-1)
                if len(_y_test.shape) == 1 and _y_test.dtype in [np.int8, np.int16, np.int32, np.int64]:
                    sewk = pd.DataFrame(_y_test).groupby(0).size().sort_index()
                    self.n_tested_samples = {int(x):sewk[int(x)] for x in self.model.classes_}
            # ROC Curve plot(クラス分類時のみ行う)
            ## ラベル数分作成する
            if len(_y_test.shape) == 1 and _y_test.dtype in [np.int8, np.int16, np.int32, np.int64]:
                for _x in df_score.columns[df_score.columns.str.contains("^predict_proba_")]:
                    y_ans = _y_test
                    y_pre = df_score[_x].values
                    lavel = int(_x.split("_")[-1])
                    self.plot_roc_curve("roc_curve_"+str(lavel), (y_ans==lavel), y_pre)
        self.logger.info("END")
        return df_conf, se_eval


    # 各特長量をランダムに変更して重要度を見積もる
    ## ※個別定義targetに関してはランダムに変更することを想定しない!!
    def calc_permutation_importance(self, df, n_trial, calc_size=0.1, eval_method="roc_auc", eval_params={"pos_label":(1,0,), "eval_auc_list":[]}):
        self.logger.info("START")
        self.logger.info("df:%s, n_trial:%s, calc_size:%s, eval_method:%s, eval_params:%s", \
                         df.shape, n_trial, calc_size, eval_method, eval_params)
        self.logger.info("features:%s", self.colname_explain.shape)
        if self.model is None:
            self.logger.raise_error("model is not set.")

        # aucの場合は確立が出せるモデルであることを前提とする
        if (eval_method == "roc_auc") or (eval_method == "roc_auc_multi"):
            if is_callable(self.model, "predict_proba") == False:
                self.logger.raise_error("this model do not predict probability.")

        # 個別に追加定義したtargetがあれば作成する
        X, Y = self.preproc(df)

        # まずは、正常な状態での評価値を求める
        ## 時間短縮のためcalc_sizeを設け、1未満のサイズではその割合のデータだけ
        ## n_trial の回数まわしてscoreを計算し、2群間のt検定をとる
        score_normal_list = np.array([])
        X_index_list = []
        if calc_size >= 1:
            df_score     = self.predict(_X=(X[0] if len(X)==1 else tuple(X)), \
                                        _Y=(Y[0] if len(Y)==1 else tuple(Y)), \
                                        pred_params={"do_estimators":False})
            score_normal = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                **eval_params)
            score_normal_list = np.append(score_normal_list, score_normal)
            for i in range(n_trial):
                X_index_list.append(np.arange(X[0].shape[0]))
        else:
            # calc_sizeの割合に縮小したデータに対してn_trial回数回す
            for i in range(n_trial):
                _random_index = np.random.permutation(np.arange(X[0].shape[0]))[:int(X[0].shape[0]*calc_size)]
                X_index_list.append(_random_index.copy())
                df_score      = self.predict(_X=(X[0][_random_index] if len(X) == 1 else tuple([__X[_random_index] for __X in X])),
                                             _Y=(Y[0][_random_index] if len(Y) == 1 else tuple([__Y[_random_index] for __Y in Y])),
                                             pred_params={"do_estimators":False})
                score_normal  = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                     df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                     **eval_params)
                score_normal_list = np.append(score_normal_list, score_normal)
        self.logger.info(f"model normal score is {eval_method}={score_normal_list.mean()} +/- {score_normal_list.std()}")

        # 特徴量をランダムに変化させて重要度を計算していく
        self.df_feature_importances_modeling = pd.DataFrame(columns=["feature_name", "p_value", "t_value", \
                                                                  "score", "score_diff", "score_std"])
        ## 短縮のため、決定木モデルでimportanceが0の特徴量は事前に省く
        colname_except_list = np.array([])
        if self.df_feature_importances.shape[0] > 0:
            colname_except_list = self.df_feature_importances[self.df_feature_importances["importance"] == 0]["feature_name"].values.copy()
        for i_colname, colname in enumerate(self.colname_explain):
            index_colname = df[self.colname_explain].columns.get_indexer([colname]).min()
            self.logger.debug("step : %s, feature is shuffled : %s", i_colname, colname)
            if np.isin(colname, colname_except_list).min() == False:
                ## 短縮リストにcolnameが存在しない場合
                score_random_list = np.empty(0)
                X_colname_bk      = X[0][:, index_colname].copy() # 後で戻せるようにバックアップする
                ## ランダムに混ぜて予測させる
                for i in range(n_trial):
                    X[:, index_colname] = np.random.permutation(X_colname_bk).copy()
                    df_score = self.predict(_X=(X[0][X_index_list[i]] if len(X) == 1 else tuple([__X[X_index_list[i]] for __X in X])), \
                                            _Y=(Y[0][X_index_list[i]] if len(Y) == 1 else tuple([__Y[X_index_list[i]] for __Y in Y])), \
                                            pred_params={"do_estimators":False})
                    score    = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                    df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                    **eval_params)
                    score_random_list = np.append(score_random_list, score)

                _t, _p = np.nan, np.nan
                if calc_size >= 1:
                    # t検定により今回のスコアの分布を評価する
                    _t, _p = stats.ttest_1samp(score_random_list, score_random_list[-1])
                else:
                    # t検定(非等分散と仮定)により、ベストスコアと今回のスコアの分布を評価する
                    _t, _p = stats.ttest_ind(score_random_list, score_random_list, axis=0, equal_var=False, nan_policy='propagate')
                
                # 結果の比較(スコアは小さいほど良い)
                self.logger.info("random score: %s = %s +/- %s. p value = %s, statistic(t value) = %s, score_list:%s", \
                                 eval_method, score_random_list.mean(), score_random_list.std(), \
                                 _p, _t, score_random_list)

                # 結果の格納
                self.df_feature_importances_modeling = \
                    self.df_feature_importances_modeling.append({"feature_name":colname, "p_value":_p, "t_value":_t, \
                                                              "score":score_random_list.mean(), \
                                                              "score_diff":score_normal - score_random_list.mean(), \
                                                              "score_std":score_random_list.std()}, ignore_index=True)
                # ランダムを元に戻す
                X[0][:, index_colname] = X_colname_bk.copy()
            else:
                ## 短縮する場合
                self.logger.info("random score: omitted")
                # 結果の格納
                self.df_feature_importances_modeling = \
                    self.df_feature_importances_modeling.append({"feature_name":colname, "p_value":np.nan, "t_value":np.nan, \
                                                              "score":np.nan, "score_diff":np.nan, "score_std":np.nan}, ignore_index=True)
        self.logger.info("END")


    # Oputuna で少数なデータから(あまり時間をかけずに)ハイパーパラメータを探索する
    # さらに、既知のアルゴリズムに対する設定を予め行っておく。
    # ※色々とパラメータ固定してからoptunaさせる方がよい. 学習率とか..
    def search_hyper_params(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, tuning_eval: str, 
        study_name: str=None, storage: str=None, load_study: bool=False, n_trials: int=10, n_jobs: int=1,
        dict_param="auto", params_add: dict=None, fit_params: dict={}, eval_params: dict={},
    ):
        """
        Params::
            storage: optuna の履歴を保存する
        Usage::
            self.search_hyper_params(
                df_train, df_test, "multi_logloss", 
                n_trials=args.get("trial", int, 200),
                dict_param = {
                    "boosting_type"    :["const", "gbdt"],
                    "num_leaves"       :["const", 100],
                    "max_depth"        :["const", -1],
                    "learning_rate"    :["const", 0.1], 
                    "n_estimators"     :["const", 1000], 
                    "subsample_for_bin":["const", 200000], 
                    "objective"        :["const", "multiclass"], 
                    "class_weight"     :["const", "balanced"], 
                    "min_child_weight" :["log", 1e-3, 1000.0],
                    "min_child_samples":["int", 1,100], 
                    "subsample"        :["float", 0.01, 1.0], 
                    "colsample_bytree" :["const", 0.25], 
                    "reg_alpha"        :["const", 0.0],
                    "reg_lambda"       :["log", 1e-3, 1000.0],
                    "random_state"     :["const", 1], 
                    "n_jobs"           :["const", mymodel.n_jobs] 
                },
                fit_params={
                    "early_stopping_rounds": 10,
                    "eval_set": [(x_valid, y_valid)],
                },
            )
        """
        self.logger.info("START")
        # numpyに変換
        X, Y           = self.preproc(df_train, autofix=True, x_proc=True, y_proc=True, row_proc=True, ret_index=False)
        X_test, Y_test = self.preproc(df_test,  autofix=True, x_proc=True, y_proc=True, row_proc=True, ret_index=False)
        df_optuna, best_params = search_hyperparams_by_optuna(
            self.model, X, Y, X_test, Y_test, 
            study_name=study_name, storage=storage, load_study=load_study, n_trials=n_trials, n_jobs=n_jobs, 
            dict_param=dict_param, params_add=params_add, tuning_eval=tuning_eval,
            fit_params=fit_params, eval_params=eval_params
        )
        self.optuna_result = df_optuna.copy()
        self.optuna_params = best_params
        self.logger.info("END")


    def plot_roc_curve(self, name, y_ans, y_pred_prob):
        self.logger.info("START")
        # canvas の追加
        self.fig[name] = plt.figure(figsize=(12, 8))
        ax = self.fig[name].add_subplot(1,1,1)

        fpr, tpr, _ = roc_curve(y_ans, y_pred_prob)
        _auc = auc(fpr, tpr)

        # ROC曲線をプロット
        ax.plot(fpr, tpr, label='ROC curve (area = %.3f)'%_auc)
        ax.legend()
        ax.set_title('ROC curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True)
        self.logger.info("END")

    def save(self, dirpath: str, filename: str=None, exist_ok: bool=False, remake: bool=False, encoding: str="utf8"):
        self.logger.info("START")
        assert isinstance(dirpath, str)
        assert isinstance(exist_ok, bool)
        assert isinstance(remake, bool)
        dirpath = correct_dirpath(dirpath)
        makedirs(dirpath, exist_ok=exist_ok, remake=remake)
        if filename is None: filename = f"mlmanager_{id(self)}.pickle"
        self.logger.info(f"save file: {dirpath + filename}.")
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
    if os.path.exists(f"{filepath}.log"):
        logger.info(f"load log file: {filepath}.log")
        with open(f"{filepath}.log", mode='r') as f:
            manager.logger.internal_stream.write(f.read())
    logger.info("END")
    return manager