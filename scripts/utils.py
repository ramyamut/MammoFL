import torch

def mse_loss(output, target):
    return torch.mean((output.squeeze() - target.squeeze())**2)

def mae_loss(output, target):
    return torch.mean(output.squeeze() - target.squeeze())

def dice(output, target):
    b = output.size(0)
    smooth = 1e-7
    iflat = output.reshape(b, -1)
    tflat = target.reshape(b, -1)
    intersection = torch.sum((iflat * tflat), dim=1)
    score = (2.0 * intersection + smooth) / (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + smooth)
    return torch.mean(score)

def dice_loss(output, target):
    b = output.size(0)
    smooth = 1e-7
    iflat = output.reshape(b, -1)
    tflat = target.reshape(b, -1)
    intersection = torch.sum((iflat * tflat), dim=1)
    score = (2.0 * intersection + smooth) / (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + smooth)
    loss = 1 - score
    return torch.mean(loss)