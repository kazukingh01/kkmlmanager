import os, shutil, pickle, base64, importlib, datetime
PICKLE_PROTOCOL = 5


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

def encode_object(_o: object, mode: int=0, savedir: str=None) -> str:
    """
    mode:
        0: base64 encoding
        1: save to file
        2: only class name (no save object)
    """
    assert isinstance(mode, int) and mode in [0, 1, 2]
    assert isinstance(savedir, (str, type(None)))
    if mode in [0, 1]:
        if not isinstance(_o, type) and hasattr(_o, "dump_with_loader"):
            output = _o.dump_with_loader()
            assert isinstance(output, dict)
            for x in ["__class__", "__loader__", "__dump_string__"]:
                assert x in output and isinstance(output[x], str)
        else:
            output = base64.b64encode(pickle.dumps(_o, protocol=PICKLE_PROTOCOL)).decode('ascii')
        if mode == 0:
            return output
        else:
            assert savedir is not None
            fname = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + "." + _o.__class__.__name__ + ".base64.txt"
            with open(os.path.join(savedir, fname), "w") as f:
                if isinstance(output, dict):
                    f.write(output["__dump_string__"])
                    output["__dump_string__"] = fname
                else:
                    f.write(output)
                    output = fname
            return output
    elif mode == 2:
        return _o.__class__.__name__

def decode_object(input_o: str | dict, basedir: str=None) -> object:
    """
    Usage::
        >>> decode_object("12345678901234567890.base64.txt")
        <object>
        >>> decode_object({"__class__": "kkmlmanager.manager.Manager", "__loader__": "load", "__dump_string__": "12345678901234567890.base64.txt"})
        <object>
    """
    assert isinstance(input_o, (str, dict))
    if isinstance(input_o, dict):
        # special loading
        for x in ["__class__", "__loader__", "__dump_string__"]: assert x in input_o
        _path, _cls = input_o["__class__"].rsplit(".", 1)
        _cls = getattr(importlib.import_module(_path), _cls)
        if len(input_o["__dump_string__"].split(".")[0]) == 20 and input_o["__dump_string__"].endswith(".base64.txt"):
            assert basedir is not None and isinstance(basedir, str)
            with open(os.path.join(basedir, input_o["__dump_string__"]), "r") as f:
                str_object = f.read()
        else:
            str_object = input_o["__dump_string__"]
        return getattr(_cls, input_o["__loader__"])(str_object)
    else:
        # general loading
        if len(input_o.split(".")[0]) == 20 and input_o.endswith(".base64.txt"):
            assert basedir is not None and isinstance(basedir, str)
            with open(os.path.join(basedir, input_o), "r") as f:
                str_object = f.read()
            return pickle.loads(base64.b64decode(str_object))
        else:
            try:
                return pickle.loads(base64.b64decode(input_o))
            except Exception as e:
                return input_o

