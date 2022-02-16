import numpy as np
import imageio
import glob
import sys
from utils import normalize_image

BREAST_IMGS = sys.argv[1]
BREAST_MASKS = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

img_paths = glob.glob(f'{BREAST_IMGS}/*.png')

for path in img_paths:
    sub_id = path.split('/')[-1]
    sub_id = sub_id.split('.png')[0]
    img = imageio.imread(path)
    mask_path = f'{BREAST_MASKS}/{sub_id}.png'
    mask = imageio.imread(mask_path)
    filtered = img * (mask / 255)
    final_img = normalize_image(filtered)
    final_img = np.expand_dims(final_img, axis=2)
    final_img = np.concatenate([final_img]*3, axis=2)
    imageio.imwrite(f'{OUTPUT_DIR}/{sub_id}.png', final_img)