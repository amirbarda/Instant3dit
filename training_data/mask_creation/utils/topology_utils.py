import torch

def consolidate_indices(f):
    """
    close gaps in index tensor
    e.g:
    [0,1,2,5,4] -> [0,1,2,4,3]
    """
    max_ele = torch.max(f)
    full_indices = torch.ones(max_ele + 1, device=f.device, dtype=torch.long) * (-1)
    full_indices[torch.unique(f)] = torch.arange(torch.unique(f).shape[0], device=f.device, dtype=torch.long)

    return full_indices[f]