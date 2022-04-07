import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# local package
from kkmlmanager.manager import MLManager


if __name__ == "__main__":
    # create dataframe
    ncols, nrows, nclass = 10, 2000, 5
    df_train = pd.DataFrame(np.random.rand(nrows, ncols), columns=[f"col_{i}" for i in range(ncols)])
    df_train["col_nan1"] = float("nan")
    df_train["col_nan2"] = float("nan")
    df_train.loc[np.random.permutation(np.arange(nrows))[:nrows//20], "col_nan2"] = np.random.rand(nrows//20)
    df_train["col_all1"] = 1
    df_train["col_sqrt"] = df_train["col_0"].pow(1/2)
    df_train["col_pw2"]  = df_train["col_0"].pow(2)
    df_train["col_pw3"]  = df_train["col_0"].pow(3)
    df_train["col_log"]  = np.log(df_train["col_0"].values)
    df_train["answer"]   = np.random.randint(0, nclass, nrows)
    df_valid = pd.DataFrame(np.random.rand(nrows, df_train.shape[1]), columns=df_train.columns.copy())
    df_valid["answer"]   = np.random.randint(0, nclass, nrows)

    # set manager
    manager = MLManager(df_train.columns[df_train.columns.str.contains("^col_")].tolist(), "answer", is_reg=False)
    # pre processing
    manager.cut_features_auto(
        df=df_train, df_test=None,
        list_proc = [
            "self.cut_features_by_variance(df, cutoff=0.9, ignore_nan=False, batch_size=128)",
            "self.cut_features_by_variance(df, cutoff=0.9, ignore_nan=True,  batch_size=128)",
            "self.cut_features_by_randomtree_importance(df, cutoff=0.9, max_iter=1, min_count=100)",
            "self.cut_features_by_correlation(df, cutoff=0.95, dtype='float32', is_gpu=False, corr_type='pearson', batch_size=1, min_n=100)",
        ]
    )
    # set model
    manager.set_model(RandomForestClassifier, bootstrap=True, n_estimators=100, max_depth=None, n_jobs=1)
    # registry proc
    manager.proc_registry(dict_proc={
        "row": [],
        "exp": [
            '"ProcAsType", np.float32, batch_size=25', 
            '"ProcToValues"', 
            '"ProcReplaceInf", posinf=float("nan"), neginf=float("nan")', 
        ],
        "ans": [
            '"ProcAsType", np.int32, n_jobs=1',
            '"ProcToValues"',
            '"ProcReshape", (-1, )',
        ]
    })
    # training
    manager.fit(df_train, df_valid=df_valid, is_proc_fit=True, is_eval_train=True)
    # cross validation
    manager.fit_cross_validation(df_train, n_split=5, n_cv=3, is_proc_fit_every_cv=True, is_save_model=True)
    # calibration
    manager.calibration(n_bins=100)
    # test evaluation
    df, se = manager.evaluate(df_valid, is_store=False)