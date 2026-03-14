import cv2
import numpy as np
import torch
from PIL import Image

from models.base import DepthModel

HF_MODEL_MAP = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2Model(DepthModel):
    def __init__(self, encoder="vitl", device=None):
        super().__init__(device)
        self.encoder = encoder

    @property
    def name(self):
        return f"Depth Anything V2 ({self.encoder})"

    def load(self):
        from transformers import pipeline

        model_id = HF_MODEL_MAP.get(self.encoder)
        if model_id is None:
            raise ValueError(
                f"Unknown encoder '{self.encoder}'. "
                f"Supported: {list(HF_MODEL_MAP.keys())}"
            )
        self.pipe = pipeline(
            "depth-estimation",
            model=model_id,
            device=self.device,
        )

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(img_rgb)
        result = self.pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float64)
        depth_map = cv2.normalize(depth, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
