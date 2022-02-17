import numpy as np
import glob
import imageio
import sys
import os

from breast_needed_functions import bring_back_images_to_orginal_size, bring_back_images_to_orginal_orientation

MASK_DIR = sys.argv[1]
PREPROC_DIR = sys.argv[2]

mask_paths = glob.glob(f'{MASK_DIR}/*.png')

for path in mask_paths:
    sub_id = path.split('/')[-1]
    sub_id = path.split('.png')[0]
    mask = imageio.imread(path) / 255
    size_csv_path = os.path.join(PREPROC_DIR, sub_id, "air_breast_mask", "fixing_ratio.csv")
    mask_postproc = bring_back_images_to_orginal_size(size_csv_path, mask)
    orig_csv_path = os.path.join(PREPROC_DIR, sub_id, "Headers.csv")
    mask_postproc = bring_back_images_to_orginal_orientation(orig_csv_path, mask_postproc)
    imageio.imwrite(path, mask_postproc)
