from abc import ABC, abstractmethod

import numpy as np
import torch


class DepthModel(ABC):

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
