# MammoDL

This repository contains the code necessary for running federated training of Deep-LIBRA, a deep-learning pipeline for breast percent density (PD) estimation from mammography. The [paper](https://www.sciencedirect.com/science/article/pii/S1361841521001845) and [code](https://github.com/CBICA/Deep-LIBRA) for the original Deep-LIBRA work are linked.

## Requirements

Python >= 3.6 is required.

Install all necessary Python packages:

`pip3 install -r requirements.txt`

Also, ensure that you are running the code on a machine with access to a GPU (cuda).

## Running Federated Training

### Setting Up Input Directories

### Running the Training Script

Run the following command:

`./pipeline/federated_wrapper.sh` path_to_dicoms_dataset1/ path_to_dicoms_dataset2`

## Running Inference/PD Calculation
