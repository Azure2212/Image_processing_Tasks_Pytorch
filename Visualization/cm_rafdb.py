import os
import sys
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import tqdm
import torch
import imgaug
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import datetime
import traceback
import shutil

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Utils/Datasets')))
from rafdb_ds import RafDataSet
from fer2013_ds import FER2013DataSet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Models/Classification_Task')))
from resnet import resnet50, resnet50_vggface2_ft

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Utils/Metrics')))
from classify_metrics import accuracy, make_batch

config_path = os.path.join(os.path.dirname(__file__), '..', 'Configs', 'config_rafdb.json')
configs = json.load(open(config_path))

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "ResNet50_CBAM", type=str, help='model2Train')
parser.add_argument('--rs-path', default= "/kaggle/working/ResnetDuck_Cbam_cuaTuan", type=str, help='trained_weight')
parser.add_argument('--statistic-path', default= '/kaggle/working/out.csv', type=str, help='trained statistic')
parser.add_argument('--dataset-used', default= 'RAFDB', type=str, help='RAFDB or FER2013')
parser.add_argument('--use-cbam', default= 1, type=int, help='use cbam= 1')
args, unknown = parser.parse_known_args()

def plot_confusion_matrix(model, testloader,title = "My model"):
    model.cuda()
    model.eval()

    correct = 0
    total = 0
    all_target = []
    all_output = []

    # test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(testloader)), total=len(testloader), leave=False):
            images, labels = testloader[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            preds = model(images).cpu()
            preds = F.softmax(preds, 1)

            # preds.shape [tta_size, 7]
            preds = torch.sum(preds, 0)
            preds = torch.argmax(preds, 0)
            preds = preds.item()
            print(labels)
            labels = labels.item()
            total += 1
            correct += preds == labels

            all_target.append(labels)
            all_output.append(preds)

    
    cf_matrix = confusion_matrix(all_target, all_output)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
    #0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

    # Create pandas dataframe
    dataframe = pd.DataFrame(cmn, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=True,cmap="Blues",fmt=".2f")
    
    plt.title(title), plt.tight_layout()
    
    plt.ylabel("True Class", fontsize=12), 
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()

    plt.savefig("DuckAttention_imagenet_RAFDB_CM.pdf")
    plt.close()

    print(classification_report(all_target, all_output, target_names=class_names))

def training_process_statistic(rs_path):
    import pandas as pd
    result = pd.read_csv(rs_path)
    plt.figure(figsize=(10, 8))
    
    # Plotting Accuracy
    plt.subplot(3, 3, 1)
    plt.plot(result['epoch'], result['accuracy'], label='Training Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(3, 3, 2)
    plt.plot(result['epoch'], result['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('val_accuracy')
    plt.legend()
    
    plt.subplot(3, 3, 3)
    plt.plot(result['epoch'], result['loss'], label='Training Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    
    plt.subplot(3, 3, 4)
    plt.plot(result['epoch'], result['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(result['epoch'], result['epoch'], label='epoch', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Epoch')
    plt.legend()
    
    plt.subplot(3, 3, 6)
    plt.plot(result['epoch'], result['learning_rate'], label='Learning Rate', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
   
if __name__ == '__main__':
    if args.dataset_used == 'RAFDB':
        test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
    else:
        test_loader_ttau = FER2013DataSet("test", configs, ttau = True, len_tta = 10) 
    model = resnet50_vggface2_ft(use_cbam = True if args.use_cbam == 1 else False)
    state = torch.load(args.rs_path)     
    model.load_state_dict(state["net"])
    print('loading completed')
    plot_confusion_matrix(model, test_loader_ttau)
    training_process_statistic(rs_path=args.statistic_path)
