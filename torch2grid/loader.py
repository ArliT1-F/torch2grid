import torch
import os

def load_torch_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Path is not a file: {path}")

    valid_extensions = ['.pt', '.pth', '.pkl']
    if not any(path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Warning: File {path} doesn't have a standart PyTorch extension (.pt, .pth, .pkl)")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except torch.serialization.pickle.UnpicklingError as e:

        try:
            print(f"Warning: weights_only=True failed, trying weights_only=False for {path}")
            obj = torch.load(path, map_location="cpu", weights_only=False) 

        except Exception as e2:
            raise RuntimeError(f"Error loading {path}: {e2}. The file may be corrupted or not a valid PyTorch file.")
    
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")
    
    return obj