import numpy as np
import pandas as pd
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
# local package
from kkmlmanager.manager import MLManager


if __name__ == "__main__":
    # load dataset
    reg     = DatasetRegistry()
    dataset = reg.create("gas-drift")
    df_train, df_test = dataset.load_data(format="polars", split_type="test", test_size=0.3)
    n_class = dataset.metadata.n_classes

    # set manager
    manager: MLManager = MLManager(
        dataset.metadata.columns_feature, dataset.metadata.columns_target,
        columns_oth=[dataset.metadata.columns_feature[0], ], is_reg=False, n_jobs=8
    )
    # test basic tree model
    manager.fit_basic_treemodel(df_train, df_valid=None, df_test=df_test, ncv=3, n_estimators=100)
    ins0 = MLManager.from_dict(manager.to_dict())
    # calibration
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=False, is_normalize=False, useerr=False, n_bins=100)
    se_eval1, df_eval = manager.evaluate(df_test, is_store=False)
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=False, is_normalize=True,  useerr=False, n_bins=100)
    se_eval2, df_eval = manager.evaluate(df_test, is_store=False)
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=False, is_normalize=False, useerr=True,  n_bins=100)
    se_eval3, df_eval = manager.evaluate(df_test, is_store=False)
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=False, is_normalize=True,  useerr=True,  n_bins=100)
    se_eval4, df_eval = manager.evaluate(df_test, is_store=False)
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=True,  is_normalize=False, useerr=False, n_bins=100, df_calib=df_test)
    se_eval5, df_eval = manager.evaluate(df_test, is_store=False)
    manager.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=True,  is_normalize=True,  useerr=False, n_bins=100, df_calib=df_test)
    se_eval6, df_eval = manager.evaluate(df_test, is_store=False)
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
    ## saved model before calibration
    ins0.set_post_model(is_cv=True, calibmodel=0, is_calib_after_cv=True,  is_normalize=True,  useerr=False, n_bins=100, df_calib=df_test)
    assert np.allclose(df_eval, ins0.evaluate(df_test, is_store=True)[-1])
    print(pd.concat([se_eval1, se_eval2, se_eval3, se_eval4, se_eval5, se_eval6], axis=1))

    # fit by GBDT
    manager.set_model(KkGBDT, n_class, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1)
    # registry proc
    manager.proc_registry()
    # training normal fit
    manager.fit_cross_validation(
        df_train, n_split=3, n_cv=3, is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit=dict(
            loss_func="multi", num_iterations=100, x_valid="_validation_x", y_valid="_validation_y",
            early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced"
        )
    )
    assert manager.model.is_softmax == True
    ins0 = MLManager.from_dict(manager.to_dict())
    manager.set_post_model(is_cv=True, calibmodel=1, is_calib_after_cv=False, is_normalize=False, useerr=False, n_bins=100)
    se_eval, df_eval = manager.evaluate(df_test, is_store=False)
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
    ## saved model before calibration
    ins0.set_post_model(is_cv=True, calibmodel=1, is_calib_after_cv=False, is_normalize=False, useerr=False, n_bins=100)
    assert np.allclose(df_eval, ins0.evaluate(df_test, is_store=True)[-1])

    # no fitting for calibration
    manager.set_post_model(
        is_cv=True, calibmodel=1, is_calib_after_cv=False, is_normalize=False, useerr=False, n_bins=100,
        kwargs_calib=dict(T=manager.model_post.models[0].calibrator.T)
    )
    se_eval, df_eval = manager.evaluate(df_test, is_store=False)
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

    manager._select_stored_dataframe("oth_V1 >= 10000")
    for icv in manager.list_cv:
        name = f"eval_valid_df_cv{icv}"
        assert getattr(manager, name).shape[0] != getattr(ins1, name).shape[0]
