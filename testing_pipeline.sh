#!/bin/bash

# Define the test images dir path and predicted masks path
test_images_dir="/home/ma22resch11003/CVIP/dataset/nnUNet_raw/Dataset001_WCEBleedGen/imagesTs"
pred_segm_dir="/home/ma22resch11003/CVIP/dataset/nnUNet_raw/Dataset001_WCEBleedGen/labelsTs"
conda_env_name="cvip"

# Activate the Conda environment
source activate $conda_env_name

# Testing classification model
python classification_test.py

# nnUnet data will stored here
nnUNet_preprocessed='/home/ma22resch11003/CVIP/nnunet_setup/nnUNet_preprocessed/'
nnUNet_results='/home/ma22resch11003/CVIP/nnunet_setup/nnUNet_results/'
nnUNet_raw='/home/ma22resch11003/CVIP/dataset/nnUNet_raw/'

# Export the nnUNet environment variables
export nnUNet_preprocessed
export nnUNet_raw
export nnUNet_results

# nnUNet args
dataset_id="001"
fold="0"
checkpoint="checkpoint_best.pth"
config="2d"

# Run nnUNetv2_predict
nnUNetv2_predict -i "$test_images_dir" -o "$pred_segm_dir" -d "$dataset_id" -f "$fold" -chk "$checkpoint" -c "$config"


