from abc import ABCMeta, abstractmethod
import gzip
from pathlib import Path
import pickle

import yaml

from mdet.utils.misc import is_list_of, is_str

FILEIO_HANDLERS = {}


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            f'handler must be a child of BaseFileHandler, not {type(handler)}')
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        FILEIO_HANDLERS[ext] = handler


def register_handler(file_formats, **kwargs):
    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap


class BaseFileHandler(metaclass=ABCMeta):
    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode='r', **kwargs):
        if 'compress' in kwargs and kwargs['compress']:
            with gzip.open(f'{filepath}.gz', mode) as f:
                return self.load_from_fileobj(f)
        else:
            with open(filepath, mode) as f:
                return self.load_from_fileobj(f)

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        if 'compress' in kwargs and kwargs['compress']:
            with gzip.open(f'{filepath}.gz', "wb") as f:
                self.dump_to_fileobj(obj, f)
        else:
            with open(filepath, mode) as f:
                self.dump_to_fileobj(obj, f)


@register_handler(['yaml', 'yml'])
class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault('Loader', yaml.FullLoader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('Dumper', yaml.Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('Dumper', yaml.Dumper)
        return yaml.dump(obj, **kwargs)


@register_handler(['pkl', 'pickle'])
class PickleHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(filepath,
                                                         mode='rb',
                                                         **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super().dump_to_path(obj, filepath, mode='wb', **kwargs)


def load(file, format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if format is None and is_str(file):
        file_name_segs = file.split('.')
        format = file_name_segs[-1]

    if format not in FILEIO_HANDLERS:
        raise TypeError(f'Unsupported format: {format}')

    handler = FILEIO_HANDLERS[format]
    if is_str(file):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dump to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if format is None:
        if is_str(file):
            format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if format not in FILEIO_HANDLERS:
        raise TypeError(f'Unsupported format: {format}')

    handler = FILEIO_HANDLERS[format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
