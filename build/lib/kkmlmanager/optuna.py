import datetime
import optuna
import pandas as pd
# local package
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "create_study",
    "load_study",
    "model_parameter_search",
]


def create_study(func, n_trials: int, storage: str=None, is_new: bool=True, name: str=None, prev_study_name: str=None, **kwargs):
    """
    Params::
        storage: postgresql://postgres:postgres@127.0.0.1:5432/optuna
    Memo::
        How to create database for PostgreSQL. # Docker command
        sudo docker exec --user=postgres postgres /usr/lib/postgresql/13/bin/dropdb optuna
        sudo docker exec --user=postgres postgres /usr/lib/postgresql/13/bin/createdb --encoding=UTF8 --locale=ja_JP.utf8 --template=template0 optuna
    """
    logger.info("START")
    assert isinstance(n_trials, int)
    assert storage is None or isinstance(storage, str)
    assert isinstance(is_new, bool)
    name = f"study_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" if name is None else name
    assert isinstance(name, str)
    study = None
    if is_new:
        if storage is not None:
            logger.info(f"create study. storage: {storage}, name: {name}")
            study = optuna.create_study(study_name=name, storage=storage, **kwargs)
        else:
            logger.info(f"load study. storage: {storage}, name: {name}")
            study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db', **kwargs)
    else:
        assert isinstance(storage, str)
        study = optuna.load_study(study_name=name, storage=storage)
    logger.info(f"start study. func: {func}, n_trials: {n_trials}")
    if prev_study_name is not None:
        assert isinstance(prev_study_name, str)
        logger.info(f"use CmaEsSampler with previous study: {prev_study_name}")
        study_prev    = optuna.load_study(storage=storage, study_name=prev_study_name)
        study.sampler = optuna.samplers.CmaEsSampler(source_trials=study_prev.trials)
    study.optimize(func, n_trials=n_trials)
    logger.info("END")
    return study

def load_study(storage: str, name: str):
    logger.info("START")
    study = optuna.load_study(study_name=name, storage=storage)
    df    = pd.DataFrame([
        [x.number, x.state, x.datetime_start, x.datetime_complete] for x in study.trials
    ], columns=["i_trial", "state", "datetime_start", "datetime_end"])
    dfwk1 = pd.DataFrame([x.params for x in study.trials])
    n_val = len(study.trials[0].values)
    dfwk2 = pd.DataFrame([(x.values if x.values is not None else [float("nan") * n_val]) for x in study.trials])
    dfwk3 = pd.DataFrame([x.user_attrs for x in study.trials])
    dfwk1.columns = [f"params_{x}" for x in dfwk1.columns]
    dfwk2.columns = [f"values_{i}" for i, _ in enumerate(dfwk2.columns)]
    dfwk3.columns = [f"usrattr_{x}" for x in dfwk3.columns]
    df = pd.concat([df, dfwk1, dfwk2, dfwk3], axis=1, ignore_index=False)
    logger.info("END")
    return df, study

def model_parameter_search(
    trial, params_search: dict=None,
    string_model: str=None, string_fit: str=None, 
    string_pred: str=None, string_eval: str=None,
    params_const: dict=None,
):
    """
    Reserved word is "_model", "_output"
    Usage::
        >>> import lightgbm as lgb
        >>> import numpy as np
        >>> train_x = np.random.rand(1000, 5)
        >>> train_y = np.random.randint(0, 5, 1000)
        >>> valid_x = np.random.rand(1000, 5)
        >>> valid_y = np.random.randint(0, 5, 1000)
        >>> from functools import partial
        >>> func = partial(
                model_parameter_search,
                params_search={
                    "min_child_weight" : "trial.suggest_loguniform('min_child_weight', 1e-8, 1e3)",
                    "subsample"        : "trial.suggest_uniform('subsample', 0.5 , 0.99)",
                    "colsample_bytree" : "trial.suggest_loguniform('colsample_bytree', 0.001, 1.0)",
                    "reg_alpha"        : "trial.suggest_loguniform('reg_alpha',  1e-8, 1e3)",
                    "reg_lambda"       : "trial.suggest_loguniform('reg_lambda', 1e-8, 1e3)",
                },
                string_model='''LGBMClassifier(
                    min_child_weight=min_child_weight, subsample=subsample, 
                    colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda                    
                )''',
                string_fit="_model.fit(train_x, train_y)",
                string_pred="_model.predict_proba(valid_x)",
                string_eval="-1 * np.log(_output[:, valid_y]).mean()",
                params_const={
                    "LGBMClassifier": lgb.LGBMClassifier, "np": np, 
                    "train_x": train_x, "train_y": train_y, "valid_x": valid_x, "valid_y": valid_y
                },
            )
        >>> create_study(func, 100, storage=None, is_new=True, name="test")
    """
    logger.info("START")
    params = {}
    for x, y in params_search.items():
        params[x] = eval(y, {}, {"trial": trial})
    params.update(params_const)
    model = eval(string_model.strip(), {}, params)
    logger.info(f"model: {model}")
    logger.info("fitting...")
    params.update({"_model": model})
    eval(string_fit.strip(), {}, params)
    logger.info("predict...")
    output = eval(string_pred.strip(), {}, params)
    params.update({"_output": output})
    logger.info("evaluate...")
    value  = eval(string_eval.strip(), {}, params)
    logger.info("END")
    return value
