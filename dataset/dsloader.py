# partition dataset loader

import os
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2

from dataset.data_utils import *

class AttnData(Dataset):
    def __init__(self, root, args, folder):
        super(AttnData, self).__init__()
        self.mode = folder
        self.folder = os.path.join(root, folder)
        self.name_list = self.collect_data_names()
        # self.name_list = self.name_list[:int(0.2*len(self.name_list))]
        # random.shuffle(self.name_list)
        print('number of'+folder+' images:', len(self.name_list))

        if isinstance(args.imgsz, int):
            self.size = [args.imgsz, args.imgsz]
        else:
            self.size = [x for x in args.imgsz]
        self.gray = args.gray
        self.aug = args.aug
        self.norm = args.norm
        self.token_size = args.token_size

    # collect all image names and store in one list
    def collect_data_names(self):
        name_list = []
        img_path = '../datasets/mots0_015/images/'+self.mode
        for img in tqdm(os.listdir(img_path)):
            name = img.split('.')[0]
            name_list.append(name)
        random.shuffle(name_list)
        return name_list

    # load data and label according to image name
    def load_data(self, name):
        img_path = os.path.join('../datasets/mots0_015/images/'+self.mode, name+'.jpg')
        # videoname = name.split('-')[0]+'-'+name.split('-')[1]
        label_path = os.path.join('../datasets/mots0_015/masks/'+self.mode, name+'.png')

        # load image
        x = Image.open(img_path)
        x = np.asarray(x)

        # load label
        y = Image.open(label_path)
        y = np.asarray(y)
        if len(y.shape) == 3:
            y = y[..., 0]
        y = y/(y.max()+10e-8) # label for each pixel would be 0-1, not binary, some values would be not 0 or 1

        h, w = int(y.shape[0]), int(y.shape[1])

        # data augmentation
        if self.aug:
            x, y = random_crop(x, y)
            x, y = random_rotate(x, y)
            x = random_light(x)

        # rgb --> grayscale
        if self.gray:
            x = c2g(x)

        # data normalization
        if self.norm and (not self.gray):
            x[..., 0] -= 123.68
            x[..., 1] -= 116.779
            x[..., 2] -= 103.939

        # resize data and label
        x = cv2.resize(x, self.size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        y = cv2.resize(y, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

        a_128 = cv2.resize(y, [self.token_size, self.token_size], interpolation=cv2.INTER_AREA).astype(np.float32)
        a_128[a_128>0] = 1
        a_64 = cv2.resize(y, [self.token_size*2, self.token_size*2], interpolation=cv2.INTER_AREA).astype(np.float32)
        a_64[a_64>0] = 1
        a_32 = cv2.resize(y, [self.token_size*4, self.token_size*4], interpolation=cv2.INTER_AREA).astype(np.float32)
        a_32[a_32>0] = 1
        y = y.reshape((1, self.size[0], self.size[1]))
        a_128 = a_128.reshape((1, self.token_size, self.token_size))
        a_64 = a_64.reshape((1, self.token_size*2, self.token_size*2))
        a_32 = a_32.reshape((1, self.token_size*4, self.token_size*4))

        if self.gray:
            x = x.reshape((1, self.size[0], self.szie[1]))
        else:
            x = np.transpose(x, (2, 0, 1))
        if self.mode == 'train':
            return x, y, a_128, a_64, a_32
        else:
            return x, y, a_128, a_64, a_32, h, w, name

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        return self.load_data(self.name_list[index])

