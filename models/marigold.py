import cv2
import numpy as np
import torch

from models.base import DepthModel


class MarigoldModel(DepthModel):

    def __init__(self, device=None):
        super().__init__(device)
        self.pipe = None

    @property
    def name(self):
        return "Marigold"

    def load(self):
        try:
            from diffusers import MarigoldDepthPipeline
        except ImportError:
            raise ImportError(
                "Marigold requires 'diffusers>=0.25'. "
                "Install with: pip install diffusers accelerate"
            )

        self.pipe = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0",
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
        ).to(self.device)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil_img = Image.fromarray(img_rgb)
        result = self.pipe(pil_img, num_inference_steps=4)
        depth = result.prediction[0, 0].cpu().numpy()
        depth_map = cv2.normalize(depth, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
