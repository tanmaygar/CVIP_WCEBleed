
import time
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from torchvision import transforms
from classification_model import ClassificationModel
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


if __name__ == "__main__":

    """ Load dataset """
    # test_x = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Images/*"))
    # test_y = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/bleeding/Annotations/*"))

    # test_x = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/images/*"))
    # test_y = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/annotation/*"))

    test_x = sorted(glob("/home/ma22resch11003/CVIP/testdata/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 1/*"))
    test_y = sorted(glob("/home/ma22resch11003/CVIP/dataset/WCEBleedGen/non-bleeding/annotation/*"))
    # exit()
    final_preds = []

    """ Hyperparameters """
    checkpoint_path = "checkpoints/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = build_unet()
    model = ClassificationModel()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    num_correct = 0
    num_incorrect = 0
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x#.split("/")[-1].split(".")[0]
        image = Image.open(name).convert('RGB')
        x = transforms.Resize(224)(image)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])(x)
        
        x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
        num_correct += np.count_nonzero(pred_y.cpu().numpy() >= 0.5)
        num_incorrect += np.count_nonzero(pred_y.cpu().numpy() < 0.5)
        file_name = name.split("/")[-1].split(".")[0]
        # print(pred_y.cpu().item())
        name_pred = "Bleeding" if pred_y.cpu() >= 0.5 else "Non-Bleeding"
        final_preds.append([file_name, name_pred])
        # exit()
        # break

    # jaccard = metrics_score[0]/len(test_x)
    # f1 = metrics_score[1]/len(test_x)
    # recall = metrics_score[2]/len(test_x)
    # precision = metrics_score[3]/len(test_x)
    # acc = metrics_score[4]/len(test_x)
    # print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
    print("1 Classified: ", num_correct/len(test_x), num_correct)
    print("0 Classified: ", num_incorrect/len(test_x), num_incorrect)
    df = pd.DataFrame(final_preds)
    df.csv("final_preds.csv", index=False)