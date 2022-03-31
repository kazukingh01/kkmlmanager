import numpy as np


__all__ = [
    "isin_compare_string",
]


def isin_compare_string(input: np.ndarray, target: np.ndarray):
    dictwk = {x:i for i, x in enumerate(target)}
    col_target = np.vectorize(lambda x: dictwk.get(x) if x in dictwk else -1)(target).astype(int)
    col_input  = np.vectorize(lambda x: dictwk.get(x) if x in dictwk else -1)(input)
    col_input  = col_input.astype(int)
    return np.isin(col_input, col_target)