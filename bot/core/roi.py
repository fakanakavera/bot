from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Scale:
    width: int
    height: int


@dataclass
class TableProfile:
    table_id: int
    roi: Rect
    scale: Scale
    landmarks: Dict[str, Tuple[int, int, int, int]]


def load_tables_config(path: str) -> List[TableProfile]:
    with open(path, 'r') as f:
        data = json.load(f)
    profiles: List[TableProfile] = []
    for t in data.get('tables', []):
        roi = Rect(**t['roi'])
        scale = Scale(**t['scale'])
        landmarks = t.get('landmarks', {})
        profiles.append(TableProfile(table_id=t['id'], roi=roi, scale=scale, landmarks=landmarks))
    return profiles
