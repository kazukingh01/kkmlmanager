import os, shutil, pickle


__all__ = [
    "makedirs",
    "check_type",
    "check_type_list",
    "correct_dirpath",
    "save_pickle",
    "load_pickle",
    "unmask_values",
    "unmask_value_isin_object",
]


def makedirs(dirpath: str, exist_ok: bool = False, remake: bool = False):
    dirpath = correct_dirpath(dirpath)
    if remake and os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok = exist_ok)

def check_type(instance: object, _type: object | list[object]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: list[object], _type: object | list[object], *args: object | list[object]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, list) or isinstance(instances, tuple):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def save_pickle(obj: object, filepath: str, *args, **kwargs):
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f, *args, **kwargs)

def load_pickle(filepath: str, *args, **kwargs) -> object:
    with open(filepath, mode='rb') as f:
        obj = pickle.load(f, *args, **kwargs)
    return obj

def unmask_values(value: object, map_dict: dict) -> object:
    assert isinstance(map_dict, dict)
    if isinstance(value, list):
        return [unmask_values(v, map_dict) for v in value]
    elif isinstance(value, tuple):
        return tuple(unmask_values(v, map_dict) for v in value)
    elif isinstance(value, dict):
        return {k: unmask_values(v, map_dict) for k, v in value.items()}
    else:
        return map_dict[value] if value in map_dict else value

def unmask_value_isin_object(value: object, map_list: list[object]) -> bool:
    assert isinstance(map_list, list)
    if isinstance(value, (tuple, list)):
        return [unmask_value_isin_object(v, map_list) for v in value]
    elif isinstance(value, dict):
        return {k: unmask_value_isin_object(v, map_list) for k, v in value.items()}
    else:
        if value in map_list:
            return True
    return False
