import torch

def dice_loss(prediction, target, smooth=1e-7):
    intersection = torch.sum(prediction * target, dim=[1,2])
    return torch.mean(1 - (2.0 * intersection + smooth) / (torch.sum(prediction, dim=[1,2]) + torch.sum(target, dim=[1,2]) + smooth))
