import sys
import os.path as osp
import tempfile
import shutil
from importlib import import_module
import mdet.utils.io as io


def load(config_path):
    full_config_path = osp.abspath(osp.expanduser(config_path))
    if full_config_path.endswith('.py'):
        return __load_py(full_config_path)
    elif full_config_path.endswith('.yaml'):
        return __load_yaml(full_config_path)
    else:
        raise IOError('Only py/yaml config file is supported')


def __load_py(full_config_path):
    file_ext = osp.splitext(full_config_path)[1]
    with tempfile.TemporaryDirectory() as tmp_config_dir:
        tmp_config_file = tempfile.NamedTemporaryFile(
            dir=tmp_config_dir, suffix=file_ext)
        tmp_config_name = osp.basename(tmp_config_file.name)
        shutil.copyfile(full_config_path, tmp_config_file.name)

        tmp_module_name = osp.splitext(tmp_config_name)[0]
        sys.path.insert(0, tmp_config_dir)
        mod = import_module(tmp_module_name)
        sys.path.pop(0)
        cfg_dict = {name: value for name,
                    value in mod.__dict__.items() if not name.startswith('_')}
        del sys.modules[tmp_module_name]
        tmp_config_file.close()
        return cfg_dict


def __load_yaml(full_config_path):
    return io.load(full_config_path, format='.yaml')
