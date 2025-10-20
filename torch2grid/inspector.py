import torch
from typing import Dict, Union, Any

# Convert any torch model or state_dict into a uniform dict of tensors.

def inspect_torch_object(obj: Union[torch.nn.Module, Dict[str, torch.Tensor]]) -> Dict[str, Any]:

    tensors = {}

    if isinstance(obj, torch.nn.Module):
        try:
            sd = obj.state_dict()
        except Exception as e:
            raise RuntimeError(f"Error extracting state_dict from model: {e}")
        
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}. Excpected nn.Module or state_dict (dict)")


    for name, tensor in sd.items():
        if torch.is_tensor(tensor):
            try:
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                tensors[name] = tensor.detach().numpy()
            except Exception as e:
                print(f"Warning: Could not convert tensor '{name}' to numpy: {e}")
                tensors[name] = None
        else:
            tensors[name] = None

    return tensors