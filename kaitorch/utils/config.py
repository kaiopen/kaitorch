from typing import Any, Dict, List, Union
from importlib import import_module
from pathlib import Path


class Configer:
    def __init__(self) -> None:
        r'''
        Read configuration from a dictionary with `merge`.

        Read configuration from a file with `merge`. Only YAML file is supported
        now. Any `'None'` in the file will be replaced by `None` automatically.
        A dictionary item ends with `_kw` will be save as a dictionary.

        `frozen` could be called to prevent data from being modified.

        ### Methods:
            - __str__
            - dict: configuration in dictionary.
            - freeze: Freeze to make sure the configuration will not be
                changed.
            - merge: Merge configuration from source.
            - print: Print the configuration.

        '''
        self.__dict__['_is_frozen'] = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._is_frozen:
            raise AttributeError(
                f'{self.__class__.__name__} object is immutable. Assignment is forbidden.'
            )
        self.__dict__[name] = value

    def __getattr__(self, name: str):
        if self._is_frozen:
            raise AttributeError(
                f'{self.__class__.__name__} object is immutable and has no attribute "{name}".'
            )

        cfg = self.__class__()
        self.__dict__[name] = cfg
        return cfg

    def __getitem__(self, name: Any):
        return self.__dict__[name]

    def __iter__(self):
        ret = []
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                ret.append((k, v))
        return iter(ret)

    def __str__(self) -> str:
        return self._str()[:-1]

    def _activate_(self, src):
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                v._activate_(src)
            elif isinstance(v, str) and v.startswith('~'):
                items = v[1:].split('.')
                obj = src[items[0]]
                for i in items[1:]:
                    obj = obj[i]
                self.__dict__[k] = obj

    def _merge_from_dict_(self, dictionary: Dict[str, Any]) -> None:
        for k, v in dictionary.items():
            if isinstance(v, dict):
                if k.endswith('_kw'):
                    self.__setattr__(k, v)
                elif k in self.__dict__:
                    self[k]._merge_from_dict_(v)
                else:
                    self.__getattr__(k)._merge_from_dict_(v)
            else:
                if isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if item == 'None':
                            v[i] = None
                elif v == 'None':
                    v = None
                self.__setattr__(k, v)

    def _merge_from_yaml_(self, path: Path) -> None:
        yaml = import_module('yaml')
        with path.open(mode='r') as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
            self._merge_from_dict_(dictionary)

    # def _merge_from_json_(self, path: Path):
    #     pass

    # def _merge_from_xml_(self, path: Path):
    #     pass

    def _str(self, indent: str = '\t', indent_level: int = 0) -> str:
        ret = ''
        ind = indent * indent_level
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, self.__class__):
                    ret += f'{ind}{key}:\n'
                    ret += value._str(indent, indent_level + 1)
                else:
                    ret += f'{ind}{key}: {value}\n'
        return ret

    def activate_(self):
        r'''Activate the configer.

        '''
        self._activate_(self)
        return self

    def dict(self) -> Dict[str, Any]:
        r'''Configuration in dictionary.

        ### Returns:
            - Configuration in dictionary.

        '''
        ret = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, self.__class__):
                    ret[key] = value.dict()
                else:
                    ret[key] = value
        return ret

    def freeze_(self):
        r'''Freeze to make sure the configuration will not be changed.

        '''
        self._is_frozen = True
        for v in self.__dict__.values():
            if isinstance(v, self.__class__):
                v.freeze_()
        return self

    def keys(self) -> List[str]:
        keys = []
        for k in self.__dict__:
            if not k.startswith('_'):
                keys.append(k)
        return keys

    def merge_(self, source: Union[Dict, str, Path]):
        r'''Merge configuration from source.

        ### Args:
            - source

        '''
        if isinstance(source, dict):
            self._merge_from_dict_(source)
            return self

        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f'{source} does not exists')
            suf = source.suffix
            if '.yaml' == suf:
                self._merge_from_yaml_(source)
            # elif suf == '.json':
            #     self._merge_from_json_(source)
            # elif suf == '.xml':
            #     self._merge_from_xml_(source)
            return self
        raise ValueError(f'Unsupported argument "{source}."')

    def print(self, indent: str = '\t', indent_level: int = 0) -> None:
        '''Print the configuration.

        ### Args:
            - indent: indent character.
            - indent_level: number of `indent` that will be printed at first.

        '''
        print(self._str(indent, indent_level)[:-1])

    def setattr(self, key: str, value):
        if self._is_frozen:
            raise AttributeError(
                f'{self.__class__.__name__} object is immutable. Assignment is forbidden.'
            )
        if not key.startswith('_') and key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise AttributeError(
                f'{self.__class__.__name__} object has no attribute named {key}.'
            )

    def values(self) -> List[Any]:
        values = []
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                values.append(v)
        return values
