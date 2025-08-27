from __future__ import annotations

from pathlib import Path

from rapidocr_onnxruntime import RapidOCR
from PIL import Image


class FastOCR:
    def __init__(self):
        self.engine = RapidOCR(use_det=True, use_cls=False, use_rec=True)
    
    def __call__(self, image: Image.Image | Path | str) -> str:
        result, _ = self.engine(image)
        if result is None:
            return ''
        return '\n'.join([line[1] for line in result])
