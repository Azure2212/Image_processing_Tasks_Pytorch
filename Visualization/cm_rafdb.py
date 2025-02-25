import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import random
import imgaug
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tqdm
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ..Utils.datasets.rafdb_ds import RafDataSet

from ..Models.resnet_v2 import  resnet50_vggface2_ft

from ..Utils.metrics.classify_metrics import accuracy, make_batch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "resnet50", type=str, help='model2Train')
parser.add_argument('--rs-dir', default= "ResnetDuck_Cbam_cuaTuan", type=str, help='rs dir in kaggle')

args, unknown = parser.parse_known_args()
path_current = os.path.abspath(globals().get("__file__","."))
script_dir  = os.path.dirname(path_current)
root_path = os.path.abspath(f"{script_dir}/../")
config_path = root_path+"/configs/config_rafdb.json"  # Adjust the path as needed

configs = json.load(open(config_path))



test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 

model = None
if args.model_name == 'resnet50_cbam_duck_pytorchcv':
    print('resnet50_cbam_duck_pytorchcv_tuan_code !')
    model = cbam_resnet50_duck()
    model.output = nn.Linear(2048, 7)
elif args.model_name == 'resnet50_vggface2':
    print('resnet50 with pre-train on vggface2(trained from cratch) was chose !')
    model = resnet50_vggface2()
elif args.model_name == 'resnet50_vggface2_ft':
    print('resnet50 with pre-train on vggface2(trained on MS1M, and then fine-tuned on VGGFace2) was chose !')
    model = resnet50_vggface2_ft(use_cbam = True)
elif args.model_name == 'resnet50_imagenet':
    print('resnet50 with pre-train on imagenet was chose !')
    model = resnet50()
elif args.model_name == 'Resnet50_in_smp':
    from sgu24project.models.segmentation_models_pytorch.model import Resnet50_in_smp
    model = Resnet50_in_smp(in_channels=3, num_seg_classes=6, num_cls_classes=7)
else:
    print('because of missing model chosen, resnet in pytorch library activated !')
    model = resnet50(pretrained = True if args.use_pretrained == 1 else False)
print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")
#state = torch.load("Rafdb_trainer_DDANet50_best_2023Jan31_05.49")
state = torch.load(args.rs_dir)     
model.load_state_dict(state["net"])


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
if __name__ == '__main__':
  plot_confusion_matrix(model, test_loader_ttau)