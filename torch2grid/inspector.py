"""
Tensor inspection utilities for extracting weights from PyTorch models.
"""

import torch
from typing import Dict, Union, Any


def inspect_torch_object(obj: Union[torch.nn.Module, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Convert any torch model or state_dict into a uniform dict of numpy arrays.
    
    This function extracts all tensor parameters from a PyTorch model or state_dict
    and converts them to numpy arrays for visualization purposes.
    
    Args:
        obj: PyTorch model (nn.Module) or state_dict (dict of tensors)
        
    Returns:
        Dictionary mapping parameter names to numpy arrays (or None for non-tensors)
        
    Raises:
        ValueError: If the object type is not supported
        RuntimeError: If there's an error converting tensors to numpy
        
    Examples:
        >>> model = torch.nn.Linear(10, 5)
        >>> tensors = inspect_torch_object(model)
        >>> print(tensors.keys())
        dict_keys(['weight', 'bias'])
        
        >>> state_dict = {'layer1.weight': torch.randn(5, 10)}
        >>> tensors = inspect_torch_object(state_dict)
        >>> print(tensors['layer1.weight'].shape)
        (5, 10)
    """
    tensors = {}

    if isinstance(obj, torch.nn.Module):
        try:
            sd = obj.state_dict()
        except Exception as e:
            raise RuntimeError(f"Error extracting state_dict from model: {e}")
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}. Expected nn.Module or state_dict (dict)")

    for name, tensor in sd.items():
        if torch.is_tensor(tensor):
            try:
                # Convert to numpy, handling different tensor types
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                tensors[name] = tensor.detach().numpy()
            except Exception as e:
                print(f"Warning: Could not convert tensor '{name}' to numpy: {e}")
                tensors[name] = None
        else:
            tensors[name] = None

    return tensors