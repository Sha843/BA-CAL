import torch
import torch.nn as nn

def bce_loss(y_pred, y_true):
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, y_true)
    return loss