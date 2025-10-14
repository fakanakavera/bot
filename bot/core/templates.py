import os
from typing import Dict, Optional, Tuple
import cv2


class TemplateLibrary:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.templates: Dict[str, Tuple[any, Optional[any]]] = {}

    def load(self, name: str, image_path: str, mask_path: Optional[str] = None) -> None:
        img = cv2.imread(os.path.join(self.base_dir, image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Template not found: {image_path}")
        mask = None
        if mask_path:
            mask = cv2.imread(os.path.join(self.base_dir, mask_path), cv2.IMREAD_GRAYSCALE)
        self.templates[name] = (img, mask)

    def get(self, name: str) -> Tuple[any, Optional[any]]:
        return self.templates[name]

    def load_dir(self, subdir: str, name_prefix: str = "", exts=(".png", ".jpg", ".jpeg")) -> None:
        dir_path = os.path.join(self.base_dir, subdir)
        if not os.path.isdir(dir_path):
            return
        for fname in sorted(os.listdir(dir_path)):
            if not fname.lower().endswith(exts):
                continue
            stem, _ = os.path.splitext(fname)
            name = f"{name_prefix}{stem}"
            self.load(name, os.path.join(subdir, fname))
