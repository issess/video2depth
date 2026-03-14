import cv2
import numpy as np
import torch

from models.base import DepthModel


class ZoeDepthModel(DepthModel):

    def __init__(self, model_type="ZoeD_NK", device=None):
        super().__init__(device)
        self.model_type = model_type

    @property
    def name(self):
        return f"ZoeDepth ({self.model_type})"

    def load(self):
        self.model = torch.hub.load('isl-org/ZoeDepth', self.model_type,
                                    pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil_img = Image.fromarray(img_rgb)
        depth = self.model.infer_pil(pil_img)
        depth_map = cv2.normalize(depth, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
