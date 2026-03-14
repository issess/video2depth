import cv2
import numpy as np
import torch

from models.base import DepthModel


class DepthAnythingV2Model(DepthModel):

    def __init__(self, encoder="vitl", device=None):
        super().__init__(device)
        self.encoder = encoder

    @property
    def name(self):
        return f"Depth Anything V2 ({self.encoder})"

    def load(self):
        self.model = torch.hub.load('depth-anything/Depth-Anything-V2',
                                    f'depth_anything_v2_{self.encoder}',
                                    trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        depth = self.model.infer_image(img_rgb)
        depth_map = cv2.normalize(depth, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
