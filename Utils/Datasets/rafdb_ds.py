import os

import sys
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Augs')))
from augmenters import Augumentations as RAFDB_aug

class RafDataSet(Dataset):
    def __init__(self, data_type, configs,  ttau = False, len_tta = 48, use_albumentation = True):
        self.use_albumentation = use_albumentation
        self.data_type = data_type
        self.configs = configs
        self.ttau = ttau
        self.len_tta = len_tta
        self.shape = (configs["image_size"], configs["image_size"])

        df = pd.read_csv(os.path.join(self.configs["raf_path"],configs["label_path"]), sep=' ', header=None,names=['name','label'])

        if data_type == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        #self.data = self.data[:100]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {data_type} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.configs["raf_path"], self.configs["image_path"], f)
            self.file_paths.append(path)

        self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Imagenet
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2675, 0.2565, 0.2761]), # VGGface2

        ]
        )
    def __len__(self):
        return len(self.file_paths)
    
    def is_ttau(self):
        return self.ttau == True

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]
#         print(image.shape)
        image = cv2.resize(image, self.shape)
        
        if self.data_type == "train":
            image = RAFDB_aug.seg_raf(image = image)
            #image = my_data_augmentation(image.copy())
        if self.data_type == "test" and self.ttau == True:
            images1 = [RAFDB_aug.seg_raftest1(image=image) for i in range(self.len_tta)]
            images2 = [RAFDB_aug.seg_raftest2(image=image) for i in range(self.len_tta)]

            images = images1 + images2
            # images = [image for i in range(self._tta_size)]
            images = list(map(self.transform, images))
            label = self.label[idx]
        
            return images, label

        image = self.transform(image)
        label = self.label[idx]
        
        return image, label