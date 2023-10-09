import os
import numpy as np
import torch
device = torch.device('cuda')
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        # Get list of all images and annotations
        self.image_paths_bleeding = [os.path.join(root_dir, 'bleeding', 'Images', fname) for fname in os.listdir(os.path.join(root_dir, 'bleeding', 'Images'))]
        self.annotation_paths_bleeding = [os.path.join(root_dir, 'bleeding', 'Annotations', fname) for fname in os.listdir(os.path.join(root_dir, 'bleeding', 'Annotations'))]

        self.image_paths_non_bleeding = [os.path.join(root_dir, 'non-bleeding', 'images', fname) for fname in os.listdir(os.path.join(root_dir, 'non-bleeding', 'images'))]
        self.annotation_paths_non_bleeding = [os.path.join(root_dir, 'non-bleeding', 'annotation', fname) for fname in os.listdir(os.path.join(root_dir, 'non-bleeding', 'annotation'))]
        
        # Aggregate paths
        self.image_paths = self.image_paths_bleeding + self.image_paths_non_bleeding
        self.annotation_paths = self.annotation_paths_bleeding + self.annotation_paths_non_bleeding
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and annotation
        image = Image.open(self.image_paths[idx]).convert('RGB')
        annotation = Image.open(self.annotation_paths[idx]).convert("L")
        
        # Convert annotation to numpy array
        annotation_array = np.array(annotation)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            annotation = transforms.ToTensor()(annotation_array)  # Convert annotation to tensor
            # annotation_transform = transforms.Compose(self.transform.transforms[1:4])
            # annotation = annotation_transform(annotation_array)
            # print("Annotation mean: ", annotation.max(), annotation.shape)
        
        # # Ensure the annotation tensor has 3 dimensions (even if the last dimension is 1)
        # if len(annotation.shape) == 2:  # if annotation is of shape [H, W]
        #     annotation = annotation.unsqueeze(0)  # Adds a channel dimension
        # print(self.annotation_paths[idx], annotation.shape)
        
        # Create label: 1 for Bleeding, 0 for Non Bleeding
        # label = 1 if 'bleeding' in self.image_paths[idx] else 0
        label = torch.tensor([0 if 'non-bleeding' in self.image_paths[idx] else 1], dtype=torch.float32)
        # print(image.shape, annotation.shape)
        
        return image, annotation, label    
