import numpy as np
import pandas as pd
import polars as pl
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
# local package
from kkmlmanager.manager import MLManager
from kkmlmanager.models import ChainModel


if __name__ == "__main__":
    # load dataset
    reg     = DatasetRegistry()
    dataset = reg.create("gas-drift")
    df_train, df_test = dataset.load_data(format="polars", split_type="test", test_size=0.3)
    n_class = dataset.metadata.n_classes

    # basic tree model
    manager1: MLManager = MLManager(
        dataset.metadata.columns_feature, dataset.metadata.columns_target,
        columns_oth=[dataset.metadata.columns_feature[0], ], is_reg=False, n_jobs=8
    )
    manager1.fit_basic_treemodel(df_train, df_valid=None, df_test=df_test, ncv=3, n_estimators=100)
    # fit by GBDT
    manager2: MLManager = MLManager(
        dataset.metadata.columns_feature, dataset.metadata.columns_target,
        columns_oth=[dataset.metadata.columns_feature[0], ], is_reg=False, n_jobs=8
    )
    manager2.set_model(KkGBDT, n_class, model_func_predict="predict", mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1)
    manager2.proc_registry()
    manager2.fit_cross_validation(
        df_train, n_split=3, n_cv=3, is_proc_fit_every_cv=True, is_save_cv_models=True,
        params_fit=dict(
            loss_func="multi", num_iterations=100, x_valid="_validation_x", y_valid="_validation_y",
            early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced"
        )
    )
    # binary model
    df_train = df_train.with_columns((pl.col("Class") == 1).cast(pl.Int32()).alias("Binary"))
    df_test  = df_test. with_columns((pl.col("Class") == 1).cast(pl.Int32()).alias("Binary"))
    manager3: MLManager = MLManager(
        dataset.metadata.columns_feature, "Binary",
        columns_oth=[dataset.metadata.columns_feature[0], ], is_reg=False, n_jobs=8
    )
    manager3.fit_basic_treemodel(df_train, df_valid=None, df_test=df_test, ncv=3, n_estimators=100)

    # chain model
    chainmodel = ChainModel(
        models=[
            {
                "model": manager1,
                "name": "modelA",
                "eval": "model.predict(input_x=input_pre, is_row=False, is_exp=True, is_ans=False, n_jobs=n_jobs)[0]",
                "shape": (-1, 6)
            },
            {
                "model": manager2,
                "name": "modelB",
                "eval": "model.predict(input_x=input_pre, is_row=False, is_exp=True, is_ans=False, n_jobs=n_jobs)[0]",
                "shape": (-1, 6)
            },
            {
                "model": manager3,
                "name": "modelC",
                "eval": "model.predict(input_x=input_pre, is_row=False, is_exp=True, is_ans=False, n_jobs=n_jobs)[0][:, -1]",
                "shape": (-1, )
            },
        ],
        eval_pre="models[0]['model'].proc_call(input, is_row=is_row, is_exp=is_exp, is_ans=is_ans)[0]",
        eval_post = """npe.normalize(npe.concatenate([
            modelA[:, 0:1] + modelB[:, 0:1],
            (modelA[:, 1:2] + modelB[:, 1:2]) * modelC.reshape(-1, 1),
            modelA[:, 2:3] + modelB[:, 2:3],
            modelA[:, 3:4] + modelB[:, 3:4],
            modelA[:, 4:5] + modelB[:, 4:5],
            modelA[:, 5:6] + modelB[:, 5:6],
        ], axis=-1), y_min=0.0, y_max=1.0, n_bins=100)""",
        func_predict="predict",
        default_params_for_predict={"is_row": False, "is_exp": True, "is_ans": False, "n_jobs": -1},
    )
    chainmodel.check_eval(np.random.rand(10))
    for x in chainmodel.models:
        x["model"].set_post_model(
            is_cv=True, calibmodel=0, is_calib_after_cv=False, list_cv=None, 
            is_normalize=True, n_bins=50, df_calib=None, useerr=True
        )
    ndf_pred = chainmodel.predict(df_test, is_row=False, is_exp=True, is_ans=True, n_jobs=-1)