import pandas as pd
import numpy as np
import cv2
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

def normalize_image(IMG):
    min = IMG.min()
    max = IMG.max()
    out = (IMG - min) / (max - min)
    out = out * 255
    try:
        out[out>255] = 255
    except:
        out = out
    try:
        out[out<0] = 0
    except:
        out = out
    out = out.astype('uint8')
    return out

def fix_ratio(IMG, height, width, method="area"):
    Flag = 0
    if np.array_equal(IMG, IMG.astype(bool)):
        IMG = IMG.astype("uint8")
        Flag = 1

    MIN = IMG.min()

    if IMG.shape[0] > IMG.shape[1]:
        IMG = np.concatenate((IMG, np.ones([IMG.shape[0],
                            IMG.shape[0]-IMG.shape[1]])*MIN), axis=1)
    else:
        IMG = np.concatenate((IMG, np.ones([IMG.shape[1]-IMG.shape[0],
                            IMG.shape[1]])*MIN), axis=0)

    if method=="area":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_AREA)
    if method=="linear":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_LINEAR)
    if method=="cubic":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_CUBIC)
    if method=="nearest":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_NEAREST)
    elif method=="lanc":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_LANCZOS4)

    return(IMG)

def object_oriented_preprocessing(MASK, sub, csv_path):
    df = pd.read_csv(csv_path)
    if 'FieldOfViewHorizontalFlip' in df.columns and df['FieldOfViewHorizontalFlip'][0]=='YES':
        if df['ImageLaterality'][0] == 'L':
            MASK=np.fliplr(MASK)
    else:
        if df['ImageLaterality'][0] == 'R':
            MASK=np.fliplr(MASK)
    return MASK
