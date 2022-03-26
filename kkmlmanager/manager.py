import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# local package
from kkutils.lib.ml.procreg import ProcRegistry
from kkutils.lib.ml.calib import Calibrater
import kkutils.lib.ml.procs as kkpr
from kkutils.util.ml.feature import search_features_by_correlation, search_features_by_variance
from kkutils.util.ml.eval import is_classification_model, evalate, eval_classification_model, eval_regressor_model, predict_detail
from kkutils.util.ml.hypara import search_hyperparams_by_optuna
from kkutils.util.ml.utils import calc_randomtree_importance, calc_parallel_mutual_information, split_data_balance, conv_validdata_in_fitparmas
from kkutils.util.dataframe import conv_ndarray
from kkutils.util.com import is_callable, correct_dirpath, makedirs, save_pickle, load_pickle, set_logger
_logname = __name__


__all__ = [
    "MyModel",
    "load_my_model"
]


class MLManager:
    def __init__(
        self, name: str, 
        # model parameter
        colname_explain: np.ndarray, colname_answer: np.ndarray, colname_other: np.ndarray = None,
        # model
        model=None, 
        # common parameter
        outdir: str="./output/", random_seed: int=1, n_jobs: int=1, log_level :str="info"
    ):
        self.logger = set_logger(_logname + "." + name, log_level=log_level, internal_log=True)
        self.logger.debug("START")
        self.name        = name
        self.outdir      = correct_dirpath(outdir)
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self.colname_explain       = conv_ndarray(colname_explain)
        self.colname_explain_hist  = []
        self.colname_answer        = conv_ndarray([colname_answer]) if isinstance(colname_answer, str) else conv_ndarray(colname_answer)
        self.colname_other         = conv_ndarray(colname_other) if colname_other is not None else np.array([])
        self.df_correlation                     = pd.DataFrame()
        self.df_feature_importances             = pd.DataFrame()
        self.df_feature_importances_randomtrees = pd.DataFrame()
        self.df_feature_importances_modeling    = pd.DataFrame()
        self.df_adversarial_valid               = pd.DataFrame()
        self.df_adversarial_importances         = pd.DataFrame()
        self.model          = model
        self.is_model_fit   = False
        self.optuna         = None
        self.calibrater     = None
        self.is_calibration = False
        self.preproc        = ProcRegistry(self.colname_explain, self.colname_answer)
        self.n_trained_samples = {}
        self.n_tested_samples  = {}
        self.index_train    = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_valid    = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_valid_cv = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_test     = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.df_pred_train    = pd.DataFrame()
        self.df_pred_valid    = pd.DataFrame()
        self.df_pred_valid_cv = pd.DataFrame()
        self.df_pred_test     = pd.DataFrame()
        self.df_cm_train      = pd.DataFrame()
        self.df_cm_valid      = pd.DataFrame()
        self.df_cm_valid_cv   = pd.DataFrame()
        self.df_cm_test       = pd.DataFrame()
        self.se_eval_train    = pd.Series(dtype=object)
        self.se_eval_valid    = pd.Series(dtype=object)
        self.se_eval_valid_cv = pd.Series(dtype=object)
        self.se_eval_test     = pd.Series(dtype=object)
        self.fig = {}
        self.logger.info("create instance. name:"+name)
        # model が set されていれば初期化しておく
        if model is not None: self.set_model(model)
        self.logger.debug("END")
        

    def __del__(self):
        pass


    def set_model(self, model, **params):
        """
        モデルのセット(モデルに関わる箇所は初期化)
        Params::
            model: model
            **params: その他インスタンスに追加したい変数
        """
        self.logger.debug("START")
        self.model          = model
        self.optuna         = None
        self.calibrater     = None
        self.is_calibration = False
        self.is_model_fit   = False
        for param in params.keys():
            # 追加のpreprocessingを扱う場合など
            self.__setattr__(param, params.get(param))
        self.logger.info(f'set model: \n{self.model}')
        self.logger.debug("END")
    

    def is_classification_model(self) -> bool:
        """
        Return:: 分類モデルの場合はTrueを返却する
        """
        return is_classification_model(self.model)
    

    def update_features(self, cut_features: np.ndarray=None, alive_features: np.ndarray=None):
        self.logger.info("START")
        self.colname_explain_hist.append(self.colname_explain.copy())
        if alive_features is None:
            self.colname_explain = self.colname_explain[~np.isin(self.colname_explain, cut_features)]
        else:
            cut_features         = self.colname_explain[~np.isin(self.colname_explain, alive_features)]
            self.colname_explain = self.colname_explain[ np.isin(self.colname_explain, alive_features)]
        self.preproc.set_columns(self.colname_explain, type_proc="x")
        self.logger.info(f"cut   features :{cut_features.shape[0]        }. features...{cut_features}")
        self.logger.info(f"alive features :{self.colname_explain.shape[0]}. features...{self.colname_explain}")
        self.logger.info("END")


    def cut_features_by_variance(self, df: pd.DataFrame, cutoff: float=0.99, ignore_nan: bool=False):
        """
        各特徴量の値の重複度合いで特徴量をカットする
        Params::
            df: input DataFrame. 既に対象の特徴量だけに絞られた状態でinputする
            cutoff: 重複度合い. 0.99の場合、全体の数値の内99%が同じ値であればカットする
            ignore_nan: nanも重複度合いの一部とするかどうか
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df.shape}, cutoff:{cutoff}, ignore_nan:{ignore_nan}")
        self.logger.info(f"features:{self.colname_explain.shape}", )
        df = df[self.colname_explain] # 関数が入れ子に成りすぎてメモリが膨大になっている
        cut_features = search_features_by_variance(df, cutoff=cutoff, ignore_nan=ignore_nan, n_jobs=self.n_jobs)
        self.update_features(cut_features) # 特徴量の更新
        self.logger.info("END")


    def cut_features_by_correlation(
        self, df: pd.DataFrame=None, cutoff=0.9, ignore_nan_mode=0, 
        n_div_col=1, on_gpu_size=1, min_n_not_nan=10, _dtype: str="float16",
        is_proc: bool=True
    ):
        """
        相関係数の高い値の特徴量をカットする
        ※欠損無視(ignore_nan=True)しての計算は計算量が多くなるので注意
        Params::
            cutoff: 相関係数の閾値
            ignore_nan_mode: 計算するモード
                0: np.corrcoef で計算. nan がないならこれで良い
                1: np.corrcoef で計算. nan は平均値で埋める
                2: pandas で計算する. nan は無視して計算するが遅いので並列化している
                3: GPUを使って高速に計算する. nanは無視して計算する
            on_gpu_size: ignore_nan_mode=3のときに使う. 行列が全てGPUに乗り切らないときに、何分割するかの数字
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df.shape if df is not None else None}, cutoff:{cutoff}, ignore_nan_mode:{ignore_nan_mode}")
        self.logger.info(f"features:{self.colname_explain.shape}")
        if is_proc:
            df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
            df_corr, _ = search_features_by_correlation(
                df[self.colname_explain], cutoff=cutoff, ignore_nan_mode=ignore_nan_mode, 
                n_div_col=n_div_col, on_gpu_size=on_gpu_size, min_n_not_nan=min_n_not_nan, 
                _dtype=_dtype, n_jobs=self.n_jobs
            )
            self.df_correlation = df_corr.copy()
        if cutoff < 1.0 and cutoff > 0:
            alive_features, cut_features = self.features_by_correlation(cutoff)
            self.update_features(cut_features, alive_features=alive_features) # 特徴量の更新
        self.logger.info("END")


    def cut_features_by_random_tree_importance(
        self, df: pd.DataFrame=None, cut_ratio: float=0, sort: bool=True, calc_randomtrees: bool=False, **kwargs
    ):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if calc_randomtrees:
            if df is None: self.logger.raise_error("dataframe is None !!")
            if self.colname_answer.shape[0] > 1: self.logger.raise_error(f"answer has over 1 columns. {self.colname_answer}")
            df   = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
            proc = ProcRegistry(self.colname_explain, self.colname_answer)
            proc.register(
                [
                    kkpr.MyAsType(np.float32),
                    kkpr.MyReplaceValue(float( "inf"), float("nan")), 
                    kkpr.MyReplaceValue(float("-inf"), float("nan")),
                    kkpr.MyFillNaMinMax(),
                ], type_proc="x"
            )
            proc.register(
                [
                    kkpr.MyAsType(np.int32),
                ], type_proc="y"
            )
            proc.fit(df)
            X, Y = proc(df, autofix=True, x_proc=True, y_proc=True, row_proc=True)
            self.df_feature_importances_randomtrees = calc_randomtree_importance(
                X, Y, colname_explain=self.colname_explain, 
                is_cls_model=self.is_classification_model(), n_jobs=self.n_jobs, **kwargs
            )
        if sort:
            self.logger.info("sort features by randomtree importance.")
            self.colname_explain_hist.append(self.colname_explain.copy())
            columns = self.df_feature_importances_randomtrees["feature_name"].values.copy()
            self.colname_explain = columns[np.isin(columns, self.colname_explain)]
        if cut_ratio > 0:
            self.logger.info(f"cut features by randomtree importance. cut_ratio: {cut_ratio}")
            alive_features, cut_features = self.features_by_random_tree_importance(cut_ratio)
            self.update_features(cut_features, alive_features=alive_features) # 特徴量の更新
        self.logger.info("END")
    

    def cut_features_by_mutual_information(self, df: pd.DataFrame, calc_size: int=50, bins: int=10, base_max: int=1):
        self.logger.info("START")
        df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
        proc = ProcRegistry(self.colname_explain, self.colname_answer)
        proc.register(
            [
                kkpr.MyAsType(np.float16),
                kkpr.MyReplaceValue(float( "inf"), float("nan")), 
                kkpr.MyReplaceValue(float("-inf"), float("nan")),
                kkpr.MyMinMaxScaler(feature_range=(0, base_max - (1./bins/10.))),
            ], type_proc="x"
        )
        proc.fit(df)
        ndf_x, _ = proc(df, autofix=True, x_proc=True, y_proc=False, row_proc=False)
        df       = pd.DataFrame(ndf_x, columns=self.colname_explain)
        self.df_mutual_information = calc_parallel_mutual_information(df, n_jobs=self.n_jobs, calc_size=calc_size, bins=bins, base_max=base_max)
        self.logger.info("END")
    

    def cut_features_by_adversarial_validation(self, cutoff=0.01):
        self.logger.info("START")
        self.logger.info(f"cutoff:{cutoff}")
        self.logger.info(f"features:{self.colname_explain.shape}")
        if self.df_adversarial_importances.shape[0] == 0:
            self.logger.raise_error("Run adversarial_validation() first.")
        alive_features, cut_features = self.features_by_adversarial_validation(cutoff)
        self.update_features(cut_features, alive_features=alive_features) # 特徴量の更新
        self.logger.info("END")


    def features_by_correlation(self, cutoff: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        columns = self.df_correlation.columns.values.copy()
        alive_features = columns[(((self.df_correlation > cutoff) | (self.df_correlation < -1*cutoff)).sum(axis=0) == 0)].copy()
        cut_list       = columns[~np.isin(columns, alive_features)]
        self.logger.info("END")
        return alive_features, cut_list


    def features_by_adversarial_validation(self, cutoff: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        columns = self.df_adversarial_importances["feature_name"].values.copy()
        cut_list       = columns[(self.df_adversarial_importances["importance"] > cutoff).values]
        alive_features = columns[~np.isin(columns, cut_list)]
        self.logger.info("END")
        return alive_features, cut_list


    def features_by_random_tree_importance(self, cut_ratio: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if self.df_feature_importances_randomtrees.shape[0] == 0:
            self.logger.raise_error("df_feature_importances_randomtrees is None. You should do calc_randomtree_importance() first !!")
        _n = int(self.df_feature_importances_randomtrees.shape[0] * cut_ratio)
        alive_features = self.df_feature_importances_randomtrees.iloc[:-1*_n ]["feature_name"].values.copy()
        cut_list       = self.df_feature_importances_randomtrees.iloc[ -1*_n:]["feature_name"].values.copy()
        self.logger.info("END")
        return alive_features, cut_list


    def feature_selection_with_random_tree_importance(
        self, df_train: pd.DataFrame, df_valid: pd.DataFrame, fit_params: dict={}, tuning_eval: str="",
        cut_corr: float=0.9, imp_top: float=0.1, imp_under: float=0.1, step: float=0.1
    ):
        """
        random tree importance の Top X% と Under Y% の特徴量を mix して、どこまでが精度が下がらないかの検証を行う。
        ※経緯として、PCA特徴量を追加したら著しく精度が悪くなったので、精度が悪くならない程度の特徴量カットをしたい
        """
        self.logger.info("START")
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if self.df_correlation.shape[0] == 0:
            self.logger.raise_error("df_correlation is None. You should do cut_features_by_correlation() first !!")
        if self.df_feature_importances_randomtrees.shape[0] == 0:
            self.logger.raise_error("df_feature_importances_randomtrees is None. You should do calc_randomtree_importance() first !!")
        colname_explain_bk = self.colname_explain.copy()

        self.colname_explain = self.df_correlation.columnsc.opy()
        self.cut_features_by_correlation(None, cutoff=cut_corr, is_proc=False)
        colname_features_randomtree = self.df_feature_importances_randomtrees["feature_name"].values.copy()
        n = colname_features_randomtree.shape[0]
        colname_explain_new = np.array(colname_features_randomtree[:int(n*imp_top)].tolist() + colname_features_randomtree[-int(n*imp_under):].tolist())
        self.update_features(alive_features=colname_explain_new)
        self.preproc.fit(df_train)
        x_train, y_train = self.preproc(df_train, autofix=True)
        x_valid, y_valid = self.preproc(df_valid, autofix=True)
        fit_params = conv_validdata_in_fitparmas(fit_params.copy(), x_valid, y_valid)
        self.model.fit(X_train, Y_train, **fit_params)

        self.logger.info("END")


    def init_proc(self):
        self.logger.info("START")
        self.preproc = ProcRegistry(self.colname_explain, self.colname_answer)
        self.logger.info("END")


    def set_default_proc(self, df: pd.DataFrame):
        self.logger.info("START")
        self.init_proc()
        self.preproc.register(
            [
                kkpr.MyAsType(np.float32),
                kkpr.MyReplaceValue(float( "inf"), float("nan")), 
                kkpr.MyReplaceValue(float("-inf"), float("nan"))
            ], type_proc="x"
        )
        self.preproc.register(
            [
                (kkpr.MyAsType(np.int32) if self.is_classification_model() else kkpr.MyAsType(np.float32)),
                kkpr.MyReshape(-1),
            ], type_proc="y"
        )
        self.preproc.register(
            [
                kkpr.MyDropNa(self.colname_answer)
            ], type_proc="row"
        )
        self.preproc.fit(df)
        self.logger.info("END")


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
    

    def adversarial_validation(
            self, df_train: pd.DataFrame, df_test: pd.DataFrame, use_answer: bool=False,
            model=None, n_splits: int=5, n_estimators: int=1000
        ):
        """
        adversarial validation. テストデータのラベルを判別するための交差顕彰
        testdataかどうかを判別しやすい特徴を省いたり、test dataの分布に近いデータを選択する事に使用する
        Params::
            df_train: train data
            df_test: test data
            model: 分類モデルのインスタンス
            n_splits: 何分割交差顕彰を行うか
        """
        self.logger.info("START")
        self.logger.info(f'df_train shape: {df_train.shape}, df_test shape: {df_test.shape}')
        self.logger.info(f"features: {self.colname_explain.shape}, answer: {self.colname_answer}")
        if model is None: model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=self.n_jobs)
        self.logger.info(f'model: \n{model}')
        # 前処理登録
        proc = ProcRegistry(self.colname_explain, self.colname_answer)
        proc.register(
            [
                kkpr.MyAsType(np.float32),
                kkpr.MyReplaceValue(float( "inf"), float("nan")), 
                kkpr.MyReplaceValue(float("-inf"), float("nan")),
                kkpr.MyFillNaMinMax(add_value=0),
            ], type_proc="x"
        )
        proc.register(
            [
                kkpr.MyAsType(np.int32),
                kkpr.MyReshape(-1),
            ], type_proc="y"
        )
        proc.fit(df_train)
        # numpyに変換(前処理の結果を反映する
        X_train, Y_train = proc(df_train, autofix=True, y_proc=use_answer, x_proc=True, row_proc=True)
        X_test,  Y_test  = proc(df_test,  autofix=True, y_proc=use_answer, x_proc=True, row_proc=True)
        if use_answer:
            # データをくっつける(正解ラベルも使う)
            X_train = np.concatenate([X_train, Y_train.reshape(-1).reshape(-1, 1)], axis=1).astype(X_train.dtype) #X_trainの型でCASTする
            X_test  = np.concatenate([X_test,  Y_test .reshape(-1).reshape(-1, 1)], axis=1).astype(X_test.dtype ) #X_test の型でCASTする
        Y_train = np.concatenate([np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])], axis=0).astype(np.int32) # 先にこっちを作る
        X_train = np.concatenate([X_train, X_test], axis=0).astype(X_train.dtype) # 連結する
        ## データのスプリット.(under samplingにしておく)
        train_indexes, test_indexes = split_data_balance(Y_train, n_splits=n_splits, y_type="cls", weight="balance", is_bootstrap=False, random_seed=self.random_seed)

        # 交差検証開始
        i_split = 1
        df_score, df_importance = pd.DataFrame(), pd.DataFrame()
        for train_index, test_index in zip(train_indexes, test_indexes):
            self.logger.info(f"create model by split samples, Cross Validation : {i_split} start...")
            _X_train = X_train[train_index]
            _Y_train = Y_train[train_index]
            _X_test  = X_train[test_index]
            _Y_test  = Y_train[test_index]
            model.fit(_X_train, _Y_train)
            self.logger.info("create model by split samples, Cross Validation : %s end...", i_split)
            # 結果の格納
            dfwk = predict_detail(model, _X_test)
            dfwk["answer"] = _Y_test
            dfwk["index"]  = test_index # concat前のtrain dataのindexは意味ある
            dfwk["type"]    = "test"
            dfwk["i_split"] = i_split
            df_score = pd.concat([df_score, dfwk], axis=0, ignore_index=True, sort=False)
            i_split += 1
            # 特徴量の重要度
            if is_callable(model, "feature_importances_") == True:
                if use_answer:
                    _df = pd.DataFrame(np.array([self.colname_explain.tolist() + self.colname_answer.tolist(), model.feature_importances_]).T, columns=["feature_name","importance"])
                else:
                    _df = pd.DataFrame(np.array([self.colname_explain.tolist(), model.feature_importances_]).T, columns=["feature_name","importance"])
                _df = _df.sort_values(by="importance", ascending=False).reset_index(drop=True)
                df_importance = pd.concat([df_importance, _df.copy()], axis=0, ignore_index=True, sort=False)
        df_score["index_df"] = -1
        df_score.loc[(df_score["answer"] == 0), "index_df"] = df_train.index[df_score.loc[(df_score["answer"] == 0), "index"].values]
        df_score.loc[(df_score["answer"] == 1), "index_df"] = df_test .index[df_score.loc[(df_score["answer"] == 1), "index"].values - df_train.shape[0]]
        self.logger.info(f'{eval_classification_model(df_score, "answer", "predict", ["predict_proba_0", "predict_proba_1"])}')
        self.df_adversarial_valid = df_score.copy()
        if df_importance.shape[0] > 0:
            df_importance["importance"] = df_importance["importance"].astype(float)
            self.df_adversarial_importances = df_importance.groupby("feature_name")["importance"].mean().reset_index().sort_values("importance", ascending=False)
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


    def save(self, dir_path: str=None, name: str=None, mode: int=0, exist_ok=False, remake=False, encoding="utf-8"):
        """
        mymodel のデータを保存します
        Params::
            dir_path: 保存するpath
            mode:
                0 : 全て保存
                1 : 全体のpickleだけ保存
                2 : 全体のpickle以外を保存
                3 : 最低限のデータのみをpickleだけ保存
        """
        self.logger.info("START")
        dir_path = self.outdir if not dir_path else correct_dirpath(dir_path)
        makedirs(dir_path, exist_ok=exist_ok, remake=remake)
        name = self.name if not name else name
        if mode in [0,2]:
            # モデルを保存
            if is_callable(self.model, "dump") == True:
                ## NNだとpickleで保存すると激重いので、dump関数がればそちらを優先する
                self.model.dump(dir_path + name + ".model.pickle")
            else:
                save_pickle(self.model, dir_path + name + ".model.pickle", protocol=4)
            # モデル情報をテキストで保存
            with open(dir_path + name + ".metadata", mode='w', encoding=encoding) as f:
                f.write("colname_explain_first="+str(self.colname_explain.tolist() if len(self.colname_explain_hist) == 0 else self.colname_explain_hist[0])+"\n")
                f.write("colname_explain="      +str(self.colname_explain.tolist())+"\n")
                f.write("colname_answer='"      +str(self.colname_answer.tolist())+"'\n")
                f.write("n_trained_samples="    +str(self.n_trained_samples)+"\n")
                f.write("n_tested_samples="     +str(self.n_tested_samples)+"\n")
                for x in self.se_eval_train.index: f.write("train_"+x+"="+str(self.se_eval_train[x])+"\n")
                for x in self.se_eval_valid.index: f.write("validation_"+x+"="+str(self.se_eval_valid[x])+"\n")
                for x in self.se_eval_test. index: f.write("test_"+x+"="+str(self.se_eval_test [x])+"\n")
            # ログを保存
            with open(dir_path + name + ".log", mode='w', encoding=encoding) as f:
                f.write(self.logger.internal_stream.getvalue())
            # 画像を保存
            for _x in self.fig.keys():
                self.fig[_x].savefig(dir_path + name + "_" + _x + '.png')
            # CSVを保存
            self.df_feature_importances_randomtrees.to_csv(dir_path + name + ".df_feature_importances_randomtrees.csv", encoding=encoding)
            self.df_feature_importances_modeling   .to_csv(dir_path + name + ".df_feature_importances_modeling.csv",    encoding=encoding)
            self.df_feature_importances            .to_csv(dir_path + name + ".df_feature_importances.csv",             encoding=encoding)
            self.df_pred_train.   to_pickle(dir_path + name + ".predict_train.pickle")
            self.df_pred_valid.   to_pickle(dir_path + name + ".predict_valid.pickle")
            self.df_pred_valid_cv.to_pickle(dir_path + name + ".predict_valid_cv.pickle")
            self.df_pred_test.    to_pickle(dir_path + name + ".predict_test.pickle")
            self.df_cm_train.   to_csv(dir_path + name + ".eval_train_confusion_matrix.csv",    encoding=encoding)
            self.df_cm_valid.   to_csv(dir_path + name + ".eval_valid_confusion_matrix.csv",    encoding=encoding)
            self.df_cm_valid_cv.to_csv(dir_path + name + ".eval_valid_cv_confusion_matrix.csv", encoding=encoding)
            self.df_cm_test.    to_csv(dir_path + name + ".eval_test_confusion_matrix.csv",     encoding=encoding)
        # 全データの保存
        if mode in [0,1]:
            save_pickle(self, dir_path + name + ".pickle", protocol=4)
        if mode in [3]:
            ## 重いデータを削除する
            self.fig = {}
            self.df_correlation                     = pd.DataFrame()
            self.df_feature_importances             = pd.DataFrame()
            self.df_feature_importances_randomtrees = pd.DataFrame()
            self.df_feature_importances_modeling    = pd.DataFrame()
            self.df_adversarial_valid               = pd.DataFrame()
            self.df_adversarial_importances         = pd.DataFrame()
            self.df_pred_train    = pd.DataFrame()
            self.df_pred_valid    = pd.DataFrame()
            self.df_pred_valid_cv = pd.DataFrame()
            self.df_pred_test     = pd.DataFrame()
            self.df_cm_train      = pd.DataFrame()
            self.df_cm_valid      = pd.DataFrame()
            self.df_cm_valid_cv   = pd.DataFrame()
            self.df_cm_test       = pd.DataFrame()
            save_pickle(self, dir_path + name + ".min.pickle", protocol=4)
        self.logger.info("END")


    def load_model(self, filepath: str, mode: int=0):
        """
        モデルだけを別ファイルから読み込む
        Params::
            mode:
                0: model をそのままコピー
                1: optuna のbest params をロード
        """
        self.logger.info("START")
        self.logger.info(f"load other My model :{filepath}")
        mymodel = load_my_model(filepath)
        if   mode == 0:
            self.model = mymodel.model.copy()
        elif mode == 1:
            best_params = mymodel.optuna.best_params.copy()
            ## 現行モデルに上書き
            self.model = self.model.set_params(**best_params)
        self.logger.info(f"\n{self.model}", )
        self.logger.info("END")


def load_my_model(filepath: str, n_jobs: int=None) -> MyModel:
    """
    MyModel形式をloadする
    Params::
        filepath: 全部が保存されたpickle名か、model名までのpathを指定する
    """
    logger = set_logger(_logname + ".__main__")
    logger.info("START")
    logger.info(f"load file: {filepath}")
    mymodel = load_pickle(filepath)
    mymodel.__class__ = MyModel
    mymodel.logger = set_logger(_logname + "." + mymodel.name, log_level="info", internal_log=True)
    if isinstance(n_jobs, int): mymodel.n_jobs = n_jobs
    logger.info("END")
    return mymodel