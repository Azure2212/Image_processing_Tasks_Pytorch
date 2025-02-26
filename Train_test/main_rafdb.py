import os
import sys
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
import torch.nn as nn

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Utils/Datasets')))
from rafdb_ds import RafDataSet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Models/Classification_Task')))
from resnet import resnet50, resnet50_vggface2_ft

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Trainers')))
from rafdb_trainer_clsTask import RAFDB_Trainer

#from sgu24project.models.resnet_cbam_v5 import resnet50_co_cbam
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default= "ResNet50_CBAM", type=str, help='model2Train')
parser.add_argument('--optimizer-chose', default= "RAdam", type=str, help='optimizer you chose')
parser.add_argument('--lr-scheduler', default= "ReduceLROnPlateau", type=str, help='learning rate scheduler you chose')
parser.add_argument('--lr-value', default= 1e-3, type=float, help='learning rate initial')
parser.add_argument('--load-state-dir', default= '', type=str, help='weight2load')
parser.add_argument('--isDebug', default= 0, type=int, help='debug = 1')
parser.add_argument('--use-pretrained', default= 1, type=int, help='use pre-trained = 1')
parser.add_argument('--use-cbam', default= 1, type=int, help='use cbam= 1')
parser.add_argument('--current-epoch-num', default= 0, type=int, help='epoch start')
parser.add_argument('--max-epoch-num', default= 1000, type=int, help='Max epoch')
parser.add_argument('--freeze-cbam', default= 0, type=int, help='use Freezing')
args, unknown = parser.parse_known_args()

print(torch.__version__)

config_path = os.path.join(os.path.dirname(__file__), '..', 'Configs', 'config_rafdb.json')
configs = json.load(open(config_path))

configs["optimizer_chose"] = args.optimizer_chose
configs["lr_scheduler"] = args.lr_scheduler
configs["lr"] = args.lr_value
configs["isDebug"] = args.isDebug
configs["current_epoch_num"] = args.current_epoch_num
configs["max_epoch_num"] = args.max_epoch_num
if args.load_state_dir != '':
    configs["load_state_dir"] = args.load_state_dir

train_loader = RafDataSet( "train", configs)
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
test_loader = RafDataSet("test", configs, ttau = False, len_tta = 48) 

print(configs)

model = None
if args.model_name == 'resnet50_vggface2':
    print('resnet50 with pre-train on vggface2(trained from cratch) was chose !')
    model = resnet50_vggface2()
elif args.model_name == 'resnet50_vggface2_ft':
    print('resnet50 with pre-train on vggface2(trained on MS1M, and then fine-tuned on VGGFace2) was chose !')
    model = resnet50_vggface2_ft(pretrained = True if args.use_pretrained == 1 else False, use_cbam = True if args.use_cbam == 1 else False)


# for name, layer in model.named_children():
#     print(f"{name}: {layer}")
configs['freeze_cbam'] = True if args.freeze_cbam == 1 else False
if configs['freeze_cbam'] == True:
    print("go freeze")
    layers = [3, 4, 6, 3]  
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.Module):  # Only consider actual layers
            if name == 'fc':
                continue
            for param in layer.parameters():
                param.requires_grad = False

    # Unfreeze CBAM blocks as specified by the layers
    for i, layer_name in enumerate(layer_names):
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for idx in range(layers[i]):
                for param in layer[idx].CbamBlock.parameters():
                    param.requires_grad = True
            
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")

print(f"the number of parameter: {sum(p.numel() for p in model.parameters())}")

trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs)

trainer.Train_model()