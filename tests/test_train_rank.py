import zipfile
import pandas as pd
import polars as pl
import numpy as np
# local package
from kkmlmanager.manager import MLManager
from kkgbdt import KkGBDT


if __name__ == '__main__':
    # load dataset
    with zipfile.ZipFile('./boatrace_course.zip', 'r') as z:
        with z.open('boatrace_course.csv') as f:
            df = pd.read_csv(f, index_col=False)
    colsoth = ["race_id", "course", "move_real"]
    df      = df.iloc[:, np.where(~(df.columns.isin(colsoth)))[0].tolist() + np.where((df.columns.isin(colsoth)))[0].tolist()]
    df["answer"]  = df["course"].map({1:6, 2:5, 3:4, 4:3, 5:2, 6:1}) # Good course is given good point
    df["race_id"] = df["race_id"].astype(str)
    ndf_bool = (df["race_id"] >= "202001010000").to_numpy(dtype=bool)
    df_test  = df.loc[ndf_bool].copy()
    df_train = df.loc[df.index.isin(df_test.index) == False].copy()
    df_train = pl.from_dataframe(df_train)
    df_test  = pl.from_dataframe(df_test )

    # set manager
    manager: MLManager = MLManager(
        ["player_no", "number", "exhibition_course", "jcd", "move_exb", "is_move_up", "is_move_exb"],
        "answer", columns_oth=["race_id", "course"], is_reg=True, n_jobs=8,
    )
    # set model
    manager.set_model(KkGBDT, 1, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1, eval_at=[3, 6])
    # registry proc
    manager.proc_registry()
    # training normal fit
    manager.fit(
        df_train, df_valid=df_test, is_proc_fit=True, is_eval_train=True,
        params_fit={
            "loss_func": "lambdarank", "num_iterations": 200, "loss_func_eval": "lambdarank", 
            "x_valid": "_validation_x", "y_valid": "_validation_y", "group_train": "_train_group", "group_valid": "_valid_group" 
        }, dict_extra_cols={"race_id": "group"}
    )
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    ins = MLManager.load_from_json(manager.to_json())
    _, dfwk = ins.evaluate(df_test, is_store=True)
    assert df_eval.equals(dfwk)

    # training cross validation
    manager.fit_cross_validation(
        df_train, n_split=3, n_cv=2, cols_multilabel_split="jcd", group_split="race_id", is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit={
            "loss_func": "lambdarank", "num_iterations": 200, "loss_func_eval": "lambdarank", 
            "x_valid": "_validation_x", "y_valid": "_validation_y", "group_train": "_train_group", "group_valid": "_valid_group" 
        }, dict_extra_cols={"race_id": "group"}
    )
    manager.set_cvmodel()
    output, input_y, input_index = manager.predict(df_test, is_row=False, is_exp=True, is_ans=False)
    se_eval, df_eval = manager.evaluate(df_test, is_store=True)
    ins = MLManager.load_from_json(manager.to_json())
    _, dfwk = ins.evaluate(df_test, is_store=True)
    assert df_eval.equals(dfwk)
