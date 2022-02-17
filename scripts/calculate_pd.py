import pandas as pd
import numpy as np
import sys
import glob
import imageio
import os

BREAST_MASKS = sys.argv[1]
DENSE_MASKS = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

breast_mask_paths = glob.glob(f'{BREAST_MASKS}/*.png')

sub_ids = []
pds = []

for path in breast_mask_paths:
    sub_id = path.split('/')[-1]
    sub_id = path.split('.png')[0]
    breast_mask = imageio.imread(path) / 255
    dense_mask_path = f'{DENSE_MASKS}/{sub_id}.png'
    dense_mask = imageio.imread(dense_mask_path) / 255
    percent_density = np.sum(dense_mask) / np.sum(breast_mask) * 100
    
    sub_ids.append(sub_id)
    pds.append(percent_density)

df = pd.DataFrame({'Subject Id': sub_ids, 'Percent Density (%)': pds})
csv_path = os.path.join(OUTPUT_DIR, "pd_inference.csv")
df.to_csv(csv_path)