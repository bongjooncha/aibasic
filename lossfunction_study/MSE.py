import torch

def mean_squared_error(y_true, y_pred):
    error = (y_true - y_pred) ** 2
    return torch.mean(error)


#pytorch 사용시
import torch.nn.functional as F

def mean_squared_error_t(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)