import importlib

MODEL_REGISTRY = {
    "midas_large":       ("models.midas", "MiDaSModel", {"model_type": "DPT_Large"}),
    "midas_hybrid":      ("models.midas", "MiDaSModel", {"model_type": "DPT_Hybrid"}),
    "midas_small":       ("models.midas", "MiDaSModel", {"model_type": "MiDaS_small"}),
    "depth_anything_v2": ("models.depth_anything_v2", "DepthAnythingV2Model", {}),
    "zoedepth":          ("models.zoedepth", "ZoeDepthModel", {}),
    "marigold":          ("models.marigold", "MarigoldModel", {}),
}

DEFAULT_MODEL = "midas_large"


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    module_path, class_name, kwargs = MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def list_models():
    return sorted(MODEL_REGISTRY.keys())
