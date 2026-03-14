import cv2
import numpy as np
import torch

from models.base import DepthModel


class MiDaSModel(DepthModel):

    def __init__(self, model_type="DPT_Large", device=None):
        super().__init__(device)
        self.model_type = model_type
        self.transform = None

    @property
    def name(self):
        return f"MiDaS ({self.model_type})"

    def load(self):
        self.model = torch.hub.load('intel-isl/MiDaS', self.model_type)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        depth_map = cv2.normalize(output, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
