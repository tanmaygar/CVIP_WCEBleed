import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
device = torch.device('cuda')
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from classification_model import ClassificationModel


def get_heatmap(model, input_batch):
    output = model(input_batch)
    # print("Req: ", output)
    gradients = torch.autograd.grad(outputs=output, inputs=input_batch,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=True, retain_graph=True)[0]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.vgg.features(input_batch)
    # print(gradients.shape)
    # print(pooled_gradients.shape)
    # print(activations.shape)
    for i in range(3):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(superimposed_img, 0.5, heatmap, 0.5, 0)
    return superimposed_img


if __name__ == "__main__":

    checkpoint_path = "checkpoints/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ClassificationModel()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Load dataset """
    test_x = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Images/*"))
    test_y = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Annotations/*"))

    # test_x = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/images/*"))
    # test_y = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/annotation/*"))

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x#.split("/")[-1].split(".")[0]
        print("Name: ", name, x)
        image = Image.open(name).convert('RGB')
        x = transforms.Resize(224)(image)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])(x)
        
        x = x.unsqueeze(0)
        x = x.to(device)
        x.requires_grad = True
        superimposed_img = get_heatmap(model, x)

        plt.imshow(superimposed_img)
        print(superimposed_img.shape)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig("test.png", bbox_inches="tight", pad_inches=0)
        # time.sleep(6)
        break
