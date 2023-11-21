import os
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2

from dataset.data_utils import *

class Data(Dataset):
    def __init__(self, root, args, folder):
        super(Data, self).__init__()
        self.folder = folder
        self.root = root
        self.name_list = self.collect_data_names()
        print('number of'+folder+' images:', len(self.name_list))

        if isinstance(args.imgsz, int):
            self.size = [args.imgsz, args.imgsz]
        else:
            self.size = [x for x in args.imgsz]
        self.aug = args.aug
        self.norm = args.norm

    # collect all image names
    def collect_data_names(self):
        name_list = []
        img_folder = os.path.join(self.root, 'images/'+self.folder)
        for img in tqdm(os.listdir(img_folder)):
            name = img.split('.')[0]
            name_list.append(name)
        random.shuffle(name_list)
        return name_list

    # load data and label according to image name
    def load_data(self, name):
        img_folder = os.path.join(self.root, 'images/'+self.folder)
        label_folder = os.path.join(self.root, 'masks/'+self.folder)
        img_path = os.path.join(img_folder, name+'.jpg')
        label_path = os.path.join(label_folder, name+'.png')

        # load image
        x = Image.open(img_path)
        x = np.asarray(x)

        # load label
        y = Image.open(label_path)
        y = np.asarray(y)
        if len(y.shape) == 3:
            y = y[..., 0]
        y = y/(y.max()+10e-8)

        # data augmentation
        if self.aug:
            x, y = random_crop(x, y)
            x, y = random_rotate(x, y)
            x = random_light(x)

        # data normalization
        if self.norm:
            x[..., 0] -= 123.68
            x[..., 1] -= 116.779
            x[..., 2] -= 103.939
        x = x/(x.max()+10e-8)

        # resize data and label
        x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        y = cv2.resize(y, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

        x = np.transpose(x, (2, 0, 1))
        y = y.reshape((1, self.size[0], self.size[1]))

        if self.folder == 'train':
            return x, y
        else:
            return x, y, name

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        return self.load_data(self.name_list[index])


