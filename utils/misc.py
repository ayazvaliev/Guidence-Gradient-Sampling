import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    return total_params