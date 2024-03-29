#Binary Cross-Entropy Loss
import torch

def binary_cross_entropy_loss(y_true, y_pred):
    loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return torch.mean(loss)


import torch.nn.functional as F

def binary_cross_entropy_loss(y_true, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)