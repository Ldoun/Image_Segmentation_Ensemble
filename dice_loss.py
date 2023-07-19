import torch

def dice_loss(prediction, target, smooth=1e-7):
    intersection = torch.sum(prediction * target)
    return 1 - (2.0 * intersection + smooth) / (torch.sum(prediction) + torch.sum(target) + smooth)
