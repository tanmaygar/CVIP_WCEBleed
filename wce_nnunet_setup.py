import os
import re
import shutil
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(description="Process paths for various image and annotation directories")

    parser.add_argument("--bleeding_image_path", required=True, help="Path to bleeding image directory")
    parser.add_argument("--bleeding_ann_path", required=True, help="Path to bleeding annotation directory")
    parser.add_argument("--non_bleeding_image_path", required=True, help="Path to non-bleeding image directory")
    parser.add_argument("--non_bleeding_ann_path", required=True, help="Path to non-bleeding annotation directory")
    parser.add_argument("--nnunet_raw_path", required=True, help="Path to nnunet_raw file")

    args = parser.parse_args()
    return args

def convert_to_binary_segmentation_mask(image_path, binary_mask_path):
    """Converts a single channel PNG image to a binary segmentation mask.

    Args:
        image: A single channel PNG image.

    Returns:
        A binary segmentation mask.
    """
    image = Image.open(image_path)

    # Convert the image to grayscale.
    grayscale_image = image.convert('L')

    # Threshold the grayscale image to create a binary mask.
    binary_mask = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')

    binary_mask.save(binary_mask_path)


if __name__ == "__main__":
    args = parse_arguments()
    
    img_paths = [args.bleeding_image_path,
                 args.non_bleeding_image_path] # bleed , non-bleed

    ann_paths = [args.non_bleeding_image_path, 
                 args.non_bleeding_ann_path]
    
    nnUNet_raw = args.nnunet_raw_path

    nnunet_paths = []
    nnunet_directories = ['imagesTr', 'labelsTr_gt', 'imagesTs', 'labelsTs_gt']
    dataset_name = 'Dataset001_WCEBleedGen'
    
    for dir in nnunet_directories:
        dir_path = os.path.join(nnUNet_raw, dataset_name, dir)
        if not os.path.exists():
            os.makedirs(dir_path)
        nnunet_paths += [dir_path]
        
    for b in range(2):
        img_list = os.listdir(img_paths[b])

        for i in range(len(img_list)):
            img_name = img_list[i]
            ann_name = img_name.replace('img', 'ann')
            img_num = str(re.findall(r'\d+', img_name)[0]).zfill(4)
            bl = b                                             
            final_name = 'WCE_' + str(bl) + img_num + '_0000' + '.png'

            train_val_split = 0.8
            if i < (train_val_split*len(img_list)):
                shutil.copyfile(os.path.join(img_paths[b], img_name), os.path.join(nnunet_directories[0], final_name))
                shutil.copyfile(os.path.join(ann_paths[b], ann_name), os.path.join(nnunet_directories[1], final_name))
            else:
                shutil.copyfile(os.path.join(img_paths[b], img_name), os.path.join(nnunet_directories[2], final_name))
                shutil.copyfile(os.path.join(ann_paths[b], ann_name), os.path.join(nnunet_directories[3], final_name))

    

        labelstr = os.path.join(nnUNet_raw, dataset_name, 'labelsTr')
        labelsts = os.path.join(nnUNet_raw, dataset_name, 'labelsTs')


        ltr = os.path.join(nnUNet_raw, dataset_name, 'labelsTr_gt') 
        lts = os.path.join(nnUNet_raw, dataset_name, 'labelsTs_gt')

        for file in os.listdir(ltr):    
            convert_to_binary_segmentation_mask(os.path.join(ltr, file),
                        os.path.join(labelstr, file))

        for file in os.listdir(lts):
            convert_to_binary_segmentation_mask(os.path.join(lts, file),
                        os.path.join(labelsts, file))
            
        num_train = len(os.listdir(os.path.join(nnUNet_raw,dataset_name, 'imagesTr')))
        
        generate_dataset_json(nnUNet_raw, {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'bleed_seg': 1}, num_train, '.png', dataset_name=dataset_name)