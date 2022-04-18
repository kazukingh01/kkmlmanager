import datetime
import optuna
# local package
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "study",
]


def study(func, n_trials: int, storage: str=None, is_new: bool=True, name: str=None):
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
            study = optuna.create_study(study_name=name, storage=storage)
        else:
            logger.info(f"load study. storage: {storage}, name: {name}")
            study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db')
    else:
        assert isinstance(storage, str)
        study = optuna.load_study(study_name=name, storage=storage)
    logger.info(f"start study. func: {func}, n_trials: {n_trials}")
    study.optimize(func, n_trials=n_trials)
    logger.info("END")
    return study
