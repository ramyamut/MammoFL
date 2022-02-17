import torch
import glob
import sys
import imageio
import numpy as np
from torchvision import transforms
from models import FederatedUNet

IMAGE_DIR = sys.argv[1]
MASK_DIR = sys.argv[2]
MODEL_PATH = sys.argv[3]

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FederatedUNet().to(device)
saved_state_dict = torch.load(MODEL_PATH, map_location='cpu')['model_state_dict']
new_state_dict = {}
for k in saved_state_dict.keys():
    new_state_dict[k.replace('model.', '')] = saved_state_dict[k]
model.eval().load_state_dict(new_state_dict)

image_paths = glob.glob(f'{IMAGE_DIR}/*.png')
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for path in image_paths:
    sub_id = path.split('/')[-1]
    sub_id = path.split('.png')[0]
    img = imageio.imread(path) / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.concatenate([img]*3, axis=2)
    img_transformed = img_transform(img)
    img_transformed = img_transformed.unsqueeze(0).to(device)
    pred_mask = model(img_transformed.float()).squeeze()
    pred_mask = torch.round(pred_mask)
    pred_mask = pred_mask.cpu().detach().numpy()
    pred_mask = (pred_mask*255).astype('uint8')
    imageio.imwrite(f'{MASK_DIR}/{sub_id}.png', pred_mask)
