import pandas as pd
import polars as pl
import numpy as np
from sklearn.datasets import fetch_california_housing
# local package
from kkmlmanager.manager import MLManager
from kkgbdt import KkGBDT


if __name__ == "__main__":
    # load dataset
    data = fetch_california_housing()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df["__answer"] = data.target
    df_train = df.iloc[np.random.permutation(np.arange(df.shape[0], dtype=int))[:-(df.shape[0] // 5)]].copy()
    df_test  = df.loc[~df.index.isin(df_train.index)].copy()
    df_train = pl.from_dataframe(df_train)
    df_test  = pl.from_dataframe(df_test )
    # set manager
    manager: MLManager = MLManager(df_train.columns[:-1], "__answer", is_reg=True, n_jobs=8)
    # pre processing
    manager.cut_features_by_variance(df_train, cutoff=0.995, ignore_nan=False)
    manager.cut_features_by_variance(df_train, cutoff=0.995, ignore_nan=True)
    manager.cut_features_by_randomtree_importance(df_train, cutoff=None, max_iter=1, min_count=100, dtype="float32")
    manager.cut_features_by_adversarial_validation(df_train, df_test, cutoff=None, thre_count='mean', n_split=3, n_cv=2, dtype="float32")
    manager.cut_features_by_correlation(df_train, cutoff=None, dtype='float32', is_gpu=False, corr_type='pearson',  batch_size=2, min_n=100)
    manager.cut_features_by_correlation(df_train, cutoff=None, dtype='float32', is_gpu=False, corr_type='spearman', batch_size=2, min_n=100)
    # set model
    manager.set_model(KkGBDT, 1, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1)
    # registry proc
    manager.proc_registry()
    # training normal fit
    manager.fit(
        df_train, df_valid=df_test, is_proc_fit=True, is_eval_train=True,
        params_fit={"loss_func": "rmse", "num_iterations": 200, "x_valid": "_validation_x", "y_valid": "_validation_y"}
    )
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    print(manager.to_json(indent=4, mode=2))
    ins1 = MLManager.from_dict(manager.to_dict())
    ins2 = MLManager.from_json(manager.to_json())
    ins3 = manager.copy(is_minimum=True)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=True, is_minimum=False, is_json=True, mode=1)
    ins4 = MLManager.load(filepath="./tmp/tmp.json", n_jobs=manager.n_jobs)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=False, is_minimum=False, is_json=False)
    ins5 = MLManager.load(filepath="./tmp/tmp.pickle", n_jobs=manager.n_jobs)
    assert df_eval.equals(ins1.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins2.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins3.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins4.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins5.evaluate(df_test, is_store=True)[-1])

    # training cross validation
    mask_split = np.ones(df_train.shape[0], dtype=bool)
    mask_split[0] = False
    manager.fit_cross_validation(
        df_train, n_split=3, n_cv=2, mask_split=mask_split, is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit=f"""dict(
            loss_func=loss_func, num_iterations=num_iterations,
            x_valid=_validation_x, y_valid=_validation_y, loss_func_eval=loss_func_eval, 
            early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name, 
            categorical_features=categorical_features
        )""",
        params_fit_evaldict={
            "loss_func": "rmse", "num_iterations": 100, "loss_func_eval": "rmse",
            "early_stopping_rounds": 20, "early_stopping_name": 0,
            "categorical_features": np.where([x.startswith("Soil_Type_") for x in manager.columns])[0].tolist(),
        },
    )
    manager.set_cvmodel()
    output, input_y, input_index = manager.predict(df_test, is_row=False, is_exp=True, is_ans=False)
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    print(manager.to_json(indent=4, mode=2))
    ins1 = MLManager.from_dict(manager.to_dict())
    ins2 = MLManager.from_json(manager.to_json())
    ins3 = manager.copy(is_minimum=True)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=True, is_minimum=False, is_json=True, mode=1)
    ins4 = MLManager.load(filepath="./tmp/tmp.json", n_jobs=manager.n_jobs)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=False, is_minimum=False, is_json=False)
    ins5 = MLManager.load(filepath="./tmp/tmp.pickle", n_jobs=manager.n_jobs)
    assert df_eval.equals(ins1.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins2.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins3.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins4.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins5.evaluate(df_test, is_store=True)[-1])

    # test basic tree model
    manager.fit_basic_treemodel(df_train, df_valid=None, df_test=df_test, ncv=2)
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    print(manager.to_json(indent=4, mode=2))
    ins1 = MLManager.from_dict(manager.to_dict())
    ins2 = MLManager.from_json(manager.to_json())
    ins3 = manager.copy(is_minimum=True)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=True, is_minimum=False, is_json=True, mode=1)
    ins4 = MLManager.load(filepath="./tmp/tmp.json", n_jobs=manager.n_jobs)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=False, is_minimum=False, is_json=False)
    ins5 = MLManager.load(filepath="./tmp/tmp.pickle", n_jobs=manager.n_jobs)
    assert df_eval.equals(ins1.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins2.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins3.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins4.evaluate(df_test, is_store=True)[-1])
    assert df_eval.equals(ins5.evaluate(df_test, is_store=True)[-1])