import torch

# Convert any torch model or state_dict into a uniform dict of tensors.

def inspect_torch_object(obj):

    tensors = {}

    if isinstance(obj, torch.nn.Module):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("Unsupported object type (excpected nn.Module or state_dict)")


    for name, tensor in sd.items():
        if torch.is_tensor(tensor):
            tensors[name] = tensor.detach().cpu().numpy()
        else:
            tensors[name] = None

    return tensors