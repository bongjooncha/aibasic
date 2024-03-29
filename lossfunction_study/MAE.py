import torch

def mean_absolute_error(y_true, y_pred):
    error = torch.abs(y_true - y_pred)
    return torch.mean(error)

import torch.nn.functional as F

def mean_absolute_error(y_true, y_pred):
    return F.l1_loss(y_pred, y_true)
