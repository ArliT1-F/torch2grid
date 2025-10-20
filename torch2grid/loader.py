import torch
import os

def load_torch_model(path):
    """
    Load a PyTorch model from file.
    
    Args:
        path: Path to the model file (.pt, .pth, or .pkl)
        
    Returns:
        Loaded PyTorch object (model, state_dict, or other)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If there's an error loading the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Path is not a file: {path}")
    
    # Check file extension
    valid_extensions = ['.pt', '.pth', '.pkl']
    if not any(path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Warning: File {path} doesn't have a standard PyTorch extension (.pt, .pth, .pkl)")
    
    try:
        # Try with weights_only=True first (safer, PyTorch 2.6+ default)
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except torch.serialization.pickle.UnpicklingError as e:
        # If weights_only=True fails, try with weights_only=False (less safe but more compatible)
        try:
            print(f"Warning: weights_only=True failed, trying weights_only=False for {path}")
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e2:
            raise RuntimeError(f"Error loading {path}: {e2}. The file may be corrupted or not a valid PyTorch file.")
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")
    
    return obj