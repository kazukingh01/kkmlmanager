import numpy as np
from kkmlmanager.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "balance_sampling",
]


def balance_sampling(ndf: np.ndarray, weight: str="over"):
    logger.info("START")
    assert len(ndf.shape) in [1, 2]
    if len(ndf.shape) == 1:
        assert ndf.dtype in [int, np.int16, np.int32, np.int64]
    assert isinstance(weight, int ) or (isinstance(weight, str) and weight in ["over", "under"])
    if len(ndf.shape) == 1:
        if isinstance(weight, int):
            cls_weight = np.bincount(ndf)[weight] / np.bincount(ndf)
        elif weight == "over":  cls_weight = np.bincount(ndf).max() / np.bincount(ndf)
        elif weight == "under": cls_weight = np.bincount(ndf).min() / np.bincount(ndf)
        assert cls_weight.shape[0] == np.unique(ndf).shape[0]
    else:
        cls_weight = ndf.sum(axis=0)
    logger.info(f"type: {weight}, class weight: {cls_weight}")
    indexes = np.arange(ndf.shape[0], dtype=int)
    ndf_n   = cls_weight.astype(int)
    ndf_f   = cls_weight - cls_weight.astype(int)
    output  = []
    if len(ndf.shape) == 1:
        for i_cls in np.arange(cls_weight.shape[0], dtype=int):
            n        = ndf_n[i_cls]
            _indexes = indexes[ndf == i_cls]
            for _ in range(n):
                output.append(_indexes.copy())
            f = ndf_f[i_cls]
            output.append(np.random.permutation(_indexes)[:int(len(_indexes) * f)].copy())
    logger.info("END")
    return np.concatenate(output)
