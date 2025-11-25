import yaml
from pathlib import Path


class DotDict(dict):
    """Dict subclass that supports dot access recursively."""
    def __getattr__(self, attr):
        if attr in self:
            value = self[attr]
            if isinstance(value, dict):
                return DotDict(value)  # recursive
            return value
        raise AttributeError(f"No attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __repr__(self):
        return f'DotDict({super().__repr__()})'


class Config:
    """Singleton-style config loader."""
    _instance = None

    def __new__(cls, config_path: str | Path = 'config.yaml'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path):
        with open(config_path, 'r') as f:
            self._data = DotDict(yaml.safe_load(f))

    # Dict-style access
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    # Dot-style access
    def __getattr__(self, attr):
        return getattr(self._data, attr)

    def __setattr__(self, attr, value):
        if attr == '_data' or attr == '_instance':
            super().__setattr__(attr, value)
        else:
            setattr(self._data, attr, value)

    def __repr__(self):
        return f'Config({self._data})'

    # DB-style access
    def get(self, *keys, default=None):
        """Nested access helper, e.g., get('database', 'host')."""
        result = self._data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def get_or_raise(self, value, *keys):
        """
        Return `value` if it is not None, otherwise return the value from config at `keys`.
        Raise ValueError if both are None.

        :param value: Optional runtime value
        :param keys: Keys to access nested config
        """
        if value is not None:
            return value

        # Get from config
        config_value = self.get(*keys)
        if config_value is not None:
            return config_value

        raise ValueError(f'Missing value for {".".join(keys)} and no default provided')