#!/bin/bash

# Define shell variables
# conda environment
conda_env_conf_path="path-to-conda-env-yaml-file"
conda_env_name="cvip"

# dataset
bleeding_imag_path="/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Images"
bleeding_ann_path="/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Annotations"
non_bleeding_img_path="/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/images"
non_bleeding_ann_path="/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/annotation"
nnunet_raw_path="/path/to/nnunet_raw.txt"

# nnUnet data will stored here
nnUNet_preprocessed='/home/ma22resch11003/CVIP/nnunet_setup/nnUNet_preprocessed/'
nnUNet_trained_models='/home/ma22resch11003/CVIP/nnunet_setup/nnUNet_trained_models/'
nnUNet_results='/home/ma22resch11003/CVIP/nnunet_setup/nnUNet_results/'

# Create the Conda environment from the configuration file
conda env create -n $conda_env_name -f $conda_env_conf_path

# Activate the Conda environment
source activate $conda_env_name

# classification training
python classification_train.py

# Check if the paths exist; if not, create them
if [ ! -d "$nnUNet_preprocessed" ]; then
    mkdir -p "$nnUNet_preprocessed"
fi

if [ ! -d "$nnUNet_trained_models" ]; then
    mkdir -p "$nnUNet_trained_models"
fi

if [ ! -d "$nnUNet_results" ]; then
    mkdir -p "$nnUNet_results"
fi

# Export the nnUNet environment variables
export nnUNet_preprocessed
export nnUNet_trained_models
export nnUNet_results

# Invoke the Python script with shell variables as arguments
python parse_paths.py \
    --bleeding_image_path "$bleeding_img_path" \
    --bleeding_ann_path "$bleeding_ann_path" \
    --non_bleeding_image_path "$non_bleeding_img_path" \
    --non_bleeding_ann_path "$non_bleeding_ann_path" \
    --nnunet_raw_path "$nnunet_raw_path"

# Define dataset ID and device
dataset_id="001"
device="cuda"
config="2d"
fold="0"

# nnUNet planning and preprocessing
nnUNetv2_plan_and_preprocess -d "$dataset_id" --verify_dataset_integrity

# nnUNet training
nnUNetv2_train "$dataset_id" "$config" "$fold" -device "$device"


