from typing import Any, Dict, List, Self, Sequence, Union
from importlib import import_module
from pathlib import Path


class Configer:
    r'''Configuration from dictionaries.

    A `Configer` object can be merged with a `Configer` object or a dictionary
    from YAML files.

    Any number, string will be read and recorded as they are. All dictionaries,
    tuple and list will be resolved iteratively.

    The method `frozen_` could be called to prevent data from being modified.

    #### Methods:
    - __setattr__
    - __getattr__
    - __getitem__
    - __iter__
    - __len__
    - __str__
    - activate_: Activate references which are strings starting with a "~".
    - dict: Get the configuration in the form of a dictionary.
    - freeze_: Freeze to make sure the configuration will not be changed.
    - keys: Get keys of the configuration.
    - merge_: Merge configuration from source.
    - pop_: Delete and return an item.
    - setattr_: Update an existed configuration item.
    - values: Get values of all configuration items.

    '''
    def __init__(self) -> None:
        self.__dict__['_is_frozen'] = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable.'
                ' Assignment is forbidden.'
            )
        self.__dict__[name] = value

    def __getattr__(self, name: str):
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable and has no'
                f' attribute "{name}".'
            )

        cfg = self.__class__()
        self.__dict__[name] = cfg
        return cfg

    def __getitem__(self, name: Any) -> Any:
        return self.__dict__[name]

    def __iter__(self):
        ret = []
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                ret.append((k, v))
        return iter(ret)

    def __len__(self):
        return len(self.__dict__) - 1

    def __str__(self) -> str:
        return self._str_cfg()

    def _activate_cfg_(self, root):
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, self.__class__):
                    v._activate_cfg_(root)
                if isinstance(v, (tuple, list)):
                    self._activate_seq_(v, root)
                elif isinstance(v, str) and v.startswith('~'):
                    items = v[1:].split('.')
                    obj = root[items[0]]
                    for item in items[1:]:
                        obj = obj[item]
                    self.__dict__[k] = obj

    @classmethod
    def _activate_seq_(cls, seq, root):
        for i, v in enumerate(seq):
            if isinstance(v, cls):
                v._activate_cfg_(root)
            elif isinstance(v, (tuple, list)):
                cls._activate_seq_(v, root)
            elif isinstance(v, str) and v.startswith('~'):
                items = v[1:].split('.')
                obj = root[items[0]]
                for item in items[1:]:
                    obj = obj[item]
                seq[i] = obj

    @classmethod
    def _freeze_seq_(cls, seq: Sequence[Any]):
        for v in seq:
            if isinstance(v, cls):
                v.freeze_()
            elif isinstance(v, (tuple, list)):
                cls._freeze_seq_(v)

    def _fuse_cfg_(self):
        for k, v in self:
            if k.startswith('~') and isinstance(v, self.__class__):
                v._fuse_cfg_()
                self.__dict__.pop(k)
                self._merge_from_cfg_(v)
            elif isinstance(v, self.__class__):
                v._fuse_cfg_()
            elif isinstance(v, (tuple, list)):
                self._fuse_seq_(v)

    @classmethod
    def _fuse_seq_(cls, seq: Sequence[Any]):
        for v in seq:
            if isinstance(v, cls):
                v._fuse_cfg_()
            elif isinstance(v, (tuple, list)):
                cls._fuse_seq_(v)

    def _merge_from_cfg_(self, cfg: Self) -> Self:
        for k, v in cfg.__dict__.items():
            if not k.startswith('_'):
                self.__dict__[k] = v
        return self

    def _merge_from_dict_(self, dictionary: Dict[str, Any]) -> Self:
        for k, v in dictionary.items():
            if isinstance(v, dict):
                self.__getattr__(k)._merge_from_dict_(v)
            elif isinstance(v, (tuple, list)):
                self.__dict__[k] = self._merge_from_seq_(v)
            else:
                self.__setattr__(k, v)
        return self

    @classmethod
    def _merge_from_seq_(cls, seq: Sequence[Any]) -> Sequence[Any]:
        for i, v in enumerate(seq):
            if isinstance(v, dict):
                seq[i] = cls()._merge_from_dict_(v)
            elif isinstance(v, (tuple, list)):
                seq[i] = cls._merge_from_seq_(v)
            else:
                seq[i] = v
        return seq

    def _merge_from_yaml_(self, path: Path) -> Self:
        yaml = import_module('yaml')
        with path.open(mode='r') as f:
            return self._merge_from_dict_(yaml.load(f, Loader=yaml.FullLoader))

    def _merge_from_json_(self, path: Path):
        pass

    def _merge_from_xml_(self, path: Path):
        pass

    @classmethod
    def _seq(cls, seq: Sequence[Any]):
        for i, v in enumerate(seq):
            if isinstance(v, cls):
                seq[i] = v.dict()
            elif isinstance(v, (tuple, list)):
                seq[i] = cls._seq(v)
        return seq

    def _str_cfg(self, indent: str = '  ', indent_level: int = 0) -> str:
        ret = ''
        ind = indent * indent_level
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, self.__class__):
                    ret += f'{ind}{k}:\n{v._str_cfg(indent, indent_level + 1)}'
                elif isinstance(v, (tuple, list)):
                    ret += f'{ind}{k}:\n{self._str_seq(v, indent, indent_level + 1)}'
                else:
                    ret += f'{ind}{k}: {v}\n'
        return ret

    @classmethod
    def _str_seq(
        cls, seq: Sequence[Any], indent: str = '  ', indent_level: int = 0
    ) -> str:
        ret = ''
        ind = indent * indent_level
        for v in seq:
            if isinstance(v, cls):
                ret += f'{ind}-\n{v._str_cfg(indent, indent_level + 1)}'
            elif isinstance(v, (tuple, list)):
                ret += f'{ind}-\n{cls._str_seq(v, indent, indent_level + 1)}'
            else:
                ret += f'{ind}- {v}\n'
        return ret

    def activate_(self) -> Self:
        r'''Activate references which are strings starting with a "~".

        #### Returns:
        - The object itself.

        '''
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable.'
            )
        self._activate_cfg_(self)
        self._fuse_cfg_()

        return self

    def dict(self) -> Dict[str, Any]:
        r'''Get the configuration in the form of a dictionary.

        #### Returns:
        - A dictionary.

        '''
        ret = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, self.__class__):
                    ret[k] = v.dict()
                elif isinstance(v, (tuple, list)):
                    ret[k] = self._seq(v)
                else:
                    ret[k] = v
        return ret

    def freeze_(self) -> Self:
        r'''Freeze to make sure the configuration will not be changed.

        NOTE: Inplace methods may change the configuration.

        #### Returns:
        - The object itself.

        '''
        if not self._is_frozen:
            self._is_frozen = True
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, self.__class__):
                    v.freeze_()
                elif isinstance(v, (tuple, list)):
                    self._freeze_seq_(v)
        return self

    def keys(self) -> List[str]:
        r'''Get keys of the configuration.

        #### Returns:
        - A list of keys.

        '''
        keys = []
        for k in self.__dict__:
            if not k.startswith('_'):
                keys.append(k)
        return keys

    def merge_(self, source: Union[Self, Dict, str, Path]) -> Self:
        r'''Merge configuration from source.

        #### Args:
        - source

        #### Returns:
        - The object itself.

        '''
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable.'
            )

        if isinstance(source, self.__class__):
            return self._merge_from_cfg_(source)

        if isinstance(source, dict):
            return self._merge_from_dict_(source)

        if isinstance(source, str):
            source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f'The source ("{source}") does not exists')
        match source.suffix[1:]:
            case 'yaml':
                return self._merge_from_yaml_(source)
            case _:
                raise ValueError(f'The source ("{source}") is not supported.')

    def pop_(self, key: str) -> Any:
        r'''Delete and return an item.

        #### Args:
        - key: the name of an attribute.

        #### Returns:
        - The value of the key.

        '''
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable.'
            )
        if key.startswith('_'):
            raise ValueError(f'The "{key}" is an invalid key.')

        return self.__dict__.pop(key)

    def setattr_(self, key: str, value: Any) -> None:
        r'''Update an existed configuration item.

        #### Args:
        - key: the name of an existed attribute.
        - value: a new value.

        '''
        if self._is_frozen:
            raise AttributeError(
                f'The {self.__class__.__name__} object is immutable.'
                ' Assignment is forbidden.'
            )
        if key.startswith('_'):
            raise ValueError(
                f'An invalid key ({key}) is given. A key should not start with'
                ' a "_".'
            )
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise ValueError(
                f'The {self.__class__.__name__} object has no attribute'
                f' "{key}".'
            )

    def values(self) -> List[Any]:
        r'''Get values of all configuration items.

        #### Returns:
        - A list of values.

        '''
        values = []
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                values.append(v)
        return values
