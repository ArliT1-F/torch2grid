import torch
import os

def load_torch_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")
    
    return obj