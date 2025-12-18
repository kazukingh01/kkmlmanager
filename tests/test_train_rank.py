import zipfile
import pandas as pd
import polars as pl
import numpy as np
from kktestdata import DatasetRegistry
# local package
from kkmlmanager.manager import MLManager
from kkgbdt import KkGBDT


if __name__ == '__main__':
    # load dataset
    reg     = DatasetRegistry()
    dataset = reg.create("boatrace_2020_2021")
    df_train, df_test = dataset.load_data(format="polars", split_type="test", test_size=0.3)
    group   = dataset.metadata.column_group
    # set manager
    manager: MLManager = MLManager(
        [x for x in dataset.metadata.columns_feature if x not in ["race_id"]], 
        dataset.metadata.columns_target, columns_oth=["race_id", "number"], is_reg=True, n_jobs=8
    )
    # set model
    manager.set_model(KkGBDT, 1, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1, eval_at=[3, 6])
    # registry proc
    manager.proc_registry()
    # training normal fit
    manager.fit(
        df_train, df_valid=df_test, is_proc_fit=True, is_eval_train=True,
        params_fit={
            "loss_func": "rank", "num_iterations": 200, "loss_func_eval": "rank", 
            "x_valid": "_validation_x", "y_valid": "_validation_y", "group_train": "_train_group", "group_valid": "_valid_group" 
        }, dict_extra_cols={group: "group"}
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
    df_train = df_train.with_columns(pl.col("race_id").cast(str).str.slice(8, 2).alias("jcd"))
    df_test  = df_test. with_columns(pl.col("race_id").cast(str).str.slice(8, 2).alias("jcd"))
    manager.fit_cross_validation(
        df_train, n_split=3, n_cv=2, cols_multilabel_split="jcd", group_split="race_id", is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit={
            "loss_func": "rank", "num_iterations": 200, "loss_func_eval": "rank", 
            "x_valid": "_validation_x", "y_valid": "_validation_y", "group_train": "_train_group", "group_valid": "_valid_group" 
        }, dict_extra_cols={group: "group"}
    )
    manager.set_post_model(is_cv=True)
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