import datetime
import optuna
import pandas as pd
# local package
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "create_study",
    "load_study",
]


def create_study(func, n_trials: int, storage: str=None, is_new: bool=True, name: str=None, **kwargs):
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
