# get arguments
ds1_dicoms=${1} # directory of dicom images for dataset 1
ds2_dicoms=${2} # directory of dicom images for dataset 2
ds1_breast_masks_orig=${3} # directory of breast mask files (png) for dataset 1
ds2_breast_masks_orig=${4} # directory of breast mask files (png) for dataset 2
ds1_dense_masks_orig=${5} # directory of dense tissue mask files (png) for dataset 1
ds2_dense_masks_orig=${6} # directory of dense tissue mask files (png) for dataset 2
output_dir=${7} # output directory that the script can write files to

# make training directories
mkdir "${output_dir}/ds1/"
mkdir "${output_dir}/ds2/"
ds1_breast_img="${output_dir}/ds1/breast_segmentation_images/"
ds1_breast_mask="${output_dir}/ds1/breast_segmentation_masks/"
ds1_dense_img="${output_dir}/ds1/density_segmentation_images/"
ds1_dense_mask="${output_dir}/ds1/density_segmentation_masks/"
ds2_breast_img="${output_dir}/ds2/breast_segmentation_images/"
ds2_breast_mask="${output_dir}/ds2/breast_segmentation_masks/"
ds2_dense_img="${output_dir}/ds2/density_segmentation_images/"
ds2_dense_mask="${output_dir}/ds2/density_segmentation_masks/"
mkdir ${ds1_breast_img}
mkdir ${ds1_breast_mask}
mkdir ${ds1_dense_img}
mkdir ${ds1_dense_mask}
mkdir ${ds2_breast_img}
mkdir ${ds2_breast_mask}
mkdir ${ds2_dense_img}
mkdir ${ds2_dense_mask}

ds1_temp="${output_dir}/ds1/temp/"
ds2_temp="${output_dir}/ds2/temp/"
mkdir ${ds1_temp}
mkdir ${ds2_temp}

# preprocess dicom images into png images as inputs to the breast segmentation model
python3 scripts/execute_libra_preprocessing.py -i ${ds1_dicoms} -o ${ds1_temp}
python3 scripts/execute_libra_preprocessing.py -i ${ds2_dicoms} -o ${ds2_temp}
cp "${ds1_temp}/air_net_data/*.png" ${ds1_breast_img}
cp "${ds2_temp}/air_net_data/*.png" ${ds2_breast_img}

# preprocess masks to match preprocessed images
python3 scripts/preprocess_masks.py ${ds1_breast_masks_orig} ${ds1_dense_masks_orig} ${ds1_breast_mask} ${ds1_dense_mask}
python3 scripts/preprocess_masks.py ${ds2_breast_masks_orig} ${ds2_dense_masks_orig} ${ds2_breast_mask} ${ds2_dense_mask}

# preprocess density segmentation model input images to only include breast
python3 scripts/preprocess_density_model_input_images.py ${ds1_breast_img} ${ds1_breast_mask} ${ds1_dense_img}
python3 scripts/preprocess_density_model_input_images.py ${ds2_breast_img} ${ds2_breast_mask} ${ds2_dense_img}

# run federated training for breast segmentation UNet
python3 scripts/federated_segmentation.py ${ds1_breast_img} ${ds1_breast_mask} ${ds2_breast_img} ${ds2_breast_mask}

# run federated training for dense tissue segmentation UNet
python3 scripts/federated_segmentation.py ${ds1_dense_img} ${ds1_dense_mask} ${ds2_dense_img} ${ds2_dense_mask}
