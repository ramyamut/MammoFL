import numpy as np
import imageio
import glob
import sys
import os
from utils import object_oriented_preprocessing, fix_ratio

MASKS_RAW = sys.argv[1]
MASKS_PREPROC = sys.argv[2]
TEMP_DIR = sys.argv[3]

mask_paths = glob.glob(f'{MASKS_RAW}/*.png')

for path in mask_paths:
    sub_id = path.split('/')[-1]
    sub_id = sub_id.split('.png')[0]
    mask = imageio.imread(path)
    csv_path = f'{TEMP_DIR}/{sub_id}/Headers.csv'
    mask = object_oriented_preprocessing(mask, sub_id, csv_path)
    mask = fix_ratio(mask, 512, 512)
    mask = (mask*255).astype('uint8')
    imageio.imwrite(f'{MASKS_PREPROC}/{sub_id}.png', mask)
