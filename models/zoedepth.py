import os
import sys
import importlib
import types
from contextlib import contextmanager
from typing import Optional

import cv2
import numpy as np
import torch

from models.base import DepthModel


@contextmanager
def prepend_sys_path(path):
    inserted = False
    if path and path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)


class ZoeDepthModel(DepthModel):
    LOCAL_HUB_REPO = os.path.expanduser("~/.cache/torch/hub/isl-org_ZoeDepth_main")

    def __init__(self, model_type="ZoeD_NK", device=None):
        super().__init__(device)
        self.model_type = model_type

    @property
    def name(self):
        return f"ZoeDepth ({self.model_type})"

    def _patch_cached_model_io(self):
        model_io = importlib.import_module("zoedepth.models.model_io")

        def patched_load_state_dict(model, state_dict):
            state_dict = state_dict.get("model", state_dict)
            do_prefix = isinstance(
                model,
                (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
            )

            state = {}
            for key, value in state_dict.items():
                if key.endswith("relative_position_index"):
                    continue
                if key.startswith("module.") and not do_prefix:
                    key = key[7:]
                if not key.startswith("module.") and do_prefix:
                    key = "module." + key
                state[key] = value

            missing, unexpected = model.load_state_dict(state, strict=False)
            unexpected = [key for key in unexpected if not key.endswith("relative_position_index")]
            if missing:
                print(f"Missing keys while loading ZoeDepth: {missing}")
            if unexpected:
                print(f"Unexpected keys while loading ZoeDepth: {unexpected}")
            return model

        model_io.load_state_dict = patched_load_state_dict

    def _patch_beit_blocks(self):
        try:
            blocks = self.model.core.core.pretrained.model.blocks
        except AttributeError:
            return

        def compat_block_forward(block, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
            attn_drop_path = getattr(block, "drop_path", getattr(block, "drop_path1", None))
            mlp_drop_path = getattr(block, "drop_path", getattr(block, "drop_path2", attn_drop_path))
            if attn_drop_path is None:
                raise AttributeError("ZoeDepth BEiT block is missing drop path layers")

            if getattr(block, "gamma_1", None) is None:
                x = x + attn_drop_path(block.attn(block.norm1(x), resolution,
                                                 shared_rel_pos_bias=shared_rel_pos_bias))
                x = x + mlp_drop_path(block.mlp(block.norm2(x)))
            else:
                x = x + attn_drop_path(block.gamma_1 * block.attn(block.norm1(x), resolution,
                                                                  shared_rel_pos_bias=shared_rel_pos_bias))
                x = x + mlp_drop_path(block.gamma_2 * block.mlp(block.norm2(x)))
            return x

        for block in blocks:
            block.forward = types.MethodType(compat_block_forward, block)

    def load(self):
        repo = self.LOCAL_HUB_REPO if os.path.exists(self.LOCAL_HUB_REPO) else "isl-org/ZoeDepth:main"
        source = "local" if repo == self.LOCAL_HUB_REPO else "github"

        if source == "local":
            with prepend_sys_path(repo):
                self._patch_cached_model_io()
                self.model = torch.hub.load(repo, self.model_type,
                                            pretrained=True, trust_repo=True,
                                            source=source, skip_validation=True)
        else:
            self.model = torch.hub.load(repo, self.model_type,
                                        pretrained=True, trust_repo=True,
                                        source=source, skip_validation=True)
        self._patch_beit_blocks()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil_img = Image.fromarray(img_rgb)
        depth = self.model.infer_pil(pil_img)
        depth_map = cv2.normalize(depth, None, 0, 1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return (depth_map * 255).astype(np.uint8)
