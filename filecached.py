"""
Basic caching of json-friendly function calls to a file

used for caching of geolocations in country_world_map.py

loads file on creation of Function() object, saves file
every time an unknown key is queried; i.e. is quite naive,
so don't try to do anything clever with it.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable


@dataclass
class Function:
    func: Callable
    file: Path
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.file = Path(self.file)
        try:
            self.data = json.loads(self.file.read_text())
        except FileNotFoundError:
            pass

    def __call__(self, key):
        try:
            return self.data[key]
        except KeyError:
            result = self.func(key)
            self.data[key] = result
            self.file.parent.mkdir(exist_ok=True, parents=True)
            self.file.write_text(json.dumps(self.data, ensure_ascii=True))
            return result
