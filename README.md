# MammoDL

This repository contains the code necessary for running federated training of Deep-LIBRA, a deep-learning pipeline for breast percent density (PD) estimation from mammography. The [paper](https://www.sciencedirect.com/science/article/pii/S1361841521001845) and [code](https://github.com/CBICA/Deep-LIBRA) for the original Deep-LIBRA work are linked.

Our base model predicts percent density by training a UNet to segment the breast from the mammogram, training a second UNet to segment the dense tissue from the image, and finally calculating the segmented breast and dense tissue areas to compute the percent density of the imaged breast. This base model improves upon the original Deep-LIBRA method by (1) getting rid of the explicit pectoral muscle segmentation step and instead incorporating pectoral muscle removal into the breast segmentation step itself, and (2) using deep learning rather than traditional machine learning methods to segment the dense tissue. Finally, we add federated learning to the training process to allow users to train models on multiple-institution datasets with privacy constraints.

## Requirements

Python >= 3.6 is required.

Install all necessary Python packages:

`pip3 install -r requirements.txt`

Also, ensure that you are running the code on a machine with access to a GPU (cuda).

## Running Federated Training

### Setting Up Input Directories

### Running the Training Script

Run the following command in the terminal, inside the MammoDL directory:

`./pipeline/federated_wrapper.sh path_to_dicoms_dataset1/ path_to_dicoms_dataset2/ path_to_breast_masks_dataset1/ path_to_breast_masks_dataset2/ path_to_dense_masks_dataset1/ path_to_dense_masks_dataset2/ output_dir/`

The last argument, the output directory, must already be created.

## Running Inference/PD Calculation
