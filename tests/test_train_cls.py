import re
import pandas as pd
import polars as pl
import numpy as np
from sklearn.datasets import fetch_covtype
# local package
from kkmlmanager.manager import MLManager
from kkgbdt import KkGBDT
from kkgbdt.loss import FocalLoss, Accuracy


if __name__ == "__main__":
    # load dataset
    data = fetch_covtype()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df["__answer"] = data.target
    df_train = df.iloc[np.random.permutation(np.arange(df.shape[0], dtype=int))[:-(df.shape[0] // 5)]].copy()
    df_test  = df.loc[~df.index.isin(df_train.index)].copy()
    df_train = pl.from_dataframe(df_train)
    df_test  = pl.from_dataframe(df_test )
    # set manager
    manager: MLManager = MLManager(df_train.columns[:-1], "__answer", is_reg=False, n_jobs=8)
    # pre processing
    manager.cut_features_by_variance(df_train, cutoff=0.995, ignore_nan=False)
    manager.cut_features_by_variance(df_train, cutoff=0.995, ignore_nan=True)
    manager.cut_features_by_randomtree_importance(df_train, cutoff=None, max_iter=1, min_count=10, dtype="float32", n_estimators=10)
    manager.cut_features_by_adversarial_validation(df_train, df_test, cutoff=None, thre_count='mean', n_split=3, n_cv=2, dtype="float32", n_estimators=10)
    manager.cut_features_by_correlation(df_train, cutoff=None, dtype='float32', is_gpu=False, corr_type='pearson',  batch_size=2, min_n=100)
    manager.cut_features_by_correlation(df_train, cutoff=None, dtype='float32', is_gpu=False, corr_type='spearman', batch_size=2, min_n=100, sample_size=100000)
    manager.initialize()
    list_proc = [
        f"self.cut_features_by_variance(cutoff=0.995, ignore_nan=False)",
        f"self.cut_features_by_variance(cutoff=0.995, ignore_nan=True )",
        f"self.cut_features_by_adversarial_validation(cutoff=1, thre_count='mean')",
        f"self.cut_features_by_randomtree_importance( cutoff=0.9)",
        f"self.cut_features_by_correlation(cutoff=0.5, corr_type='pearson')",
        f"self.cut_features_by_correlation(cutoff=0.1, corr_type='spearman')",
    ]
    manager.cut_features_auto(list_proc=list_proc)
    ins1 = MLManager.from_dict(manager.to_dict()).initialize().cut_features_auto(list_proc=list_proc)
    ins2 = MLManager.from_json(manager.to_json()).initialize().cut_features_auto(list_proc=list_proc)
    ins3 = manager.copy().initialize().cut_features_auto(list_proc=list_proc)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=True, is_minimum=False, is_json=True, mode=1)
    ins4 = MLManager.load(filepath="./tmp/tmp.json", n_jobs=manager.n_jobs).initialize().cut_features_auto(list_proc=list_proc)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=False, is_minimum=False, is_json=False)
    ins5 = MLManager.load(filepath="./tmp/tmp.pickle", n_jobs=manager.n_jobs).initialize().cut_features_auto(list_proc=list_proc)
    assert np.all(manager.columns == ins1.columns)
    assert np.all(manager.columns == ins2.columns)
    assert np.all(manager.columns == ins3.columns)
    assert np.all(manager.columns == ins4.columns)
    assert np.all(manager.columns == ins5.columns)
    assert len(manager.columns_hist) == len(ins1.columns_hist)
    assert len(manager.columns_hist) == len(ins2.columns_hist)
    assert len(manager.columns_hist) == len(ins3.columns_hist)
    assert len(manager.columns_hist) == len(ins4.columns_hist)
    assert len(manager.columns_hist) == len(ins5.columns_hist)
    for x, y in zip(manager.columns_hist, ins1.columns_hist): assert np.all(x == y)
    for x, y in zip(manager.columns_hist, ins2.columns_hist): assert np.all(x == y)
    for x, y in zip(manager.columns_hist, ins3.columns_hist): assert np.all(x == y)
    for x, y in zip(manager.columns_hist, ins4.columns_hist): assert np.all(x == y)
    for x, y in zip(manager.columns_hist, ins5.columns_hist): assert np.all(x == y)
    for x in dir(manager):
        if re.match(r"^features_", x) is not None:
            assert getattr(manager, x).equals(getattr(ins1, x))
            assert getattr(manager, x).equals(getattr(ins2, x))
            assert getattr(manager, x).equals(getattr(ins3, x))
            assert getattr(manager, x).equals(getattr(ins4, x))
            assert getattr(manager, x).equals(getattr(ins5, x))
    # set model
    n_class = df_train["__answer"].unique().len()
    manager.set_model(KkGBDT, n_class, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1)
    # registry proc
    manager.proc_registry()
    # training normal fit
    manager.fit(
        df_train, df_valid=df_test, is_proc_fit=True, is_eval_train=True,
        params_fit={"loss_func": "multiclass", "num_iterations": 20, "sample_weight": "balanced", "x_valid": "_validation_x", "y_valid": "_validation_y"}
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
    assert np.allclose(df_eval, ins1.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins2.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins3.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins4.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins5.evaluate(df_test, is_store=True)[-1])

    # training cross validation
    mask_split = np.ones(df_train.shape[0], dtype=bool)
    mask_split[0] = False
    manager.fit_cross_validation(
        df_train, n_split=3, n_cv=2, mask_split=mask_split, is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit=f"""dict(
            loss_func=loss_func, num_iterations=num_iterations,
            x_valid=_validation_x, y_valid=_validation_y, loss_func_eval=loss_func_eval, 
            early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name, 
            sample_weight="balanced", categorical_features=categorical_features
        )""",
        params_fit_evaldict={
            "loss_func": FocalLoss(n_class, gamma=0.5, dx=1e-5), "num_iterations": 20, "loss_func_eval": ["__copy__", Accuracy(top_k=2)],
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
    assert np.allclose(df_eval, ins1.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins2.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins3.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins4.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins5.evaluate(df_test, is_store=True)[-1])

    # test basic tree model
    manager.fit_basic_treemodel(df_train, df_valid=None, df_test=df_test, ncv=2, n_estimators=20)
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    print(manager.to_json(indent=4, mode=2))
    ins1 = MLManager.from_dict(manager.to_dict())
    ins2 = MLManager.from_json(manager.to_json())
    ins3 = manager.copy(is_minimum=True)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=True, is_minimum=False, is_json=True, mode=1)
    ins4 = MLManager.load(filepath="./tmp/tmp.json", n_jobs=manager.n_jobs)
    manager.save(dirpath="./tmp/", filename="tmp", is_remake=False, is_minimum=False, is_json=False)
    ins5 = MLManager.load(filepath="./tmp/tmp.pickle", n_jobs=manager.n_jobs)
    assert np.allclose(df_eval, ins1.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins2.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins3.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins4.evaluate(df_test, is_store=True)[-1])
    assert np.allclose(df_eval, ins5.evaluate(df_test, is_store=True)[-1])

    # # calibration
    # manager.calibration(is_use_valid=True, n_bins=100)
    # manager.calibration(is_use_valid=True, n_bins=100, is_binary_fit=True)
