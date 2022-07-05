import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2


class listDataset(Dataset):
    def __init__(self, root_unlabel, root_label, shape=None, shuffle=True, transform=None,
                 train=False, seen=0, batch_size=1, num_workers=4):

        if train:
            root = 1 * root_unlabel + 4 * root_label

        else:
            root = 1 * (1 * root_unlabel + 1 * root_label)
        random.shuffle(root)
        random.shuffle(root_unlabel)

        self.nSamples = len(root)
        self.lines = root
        self.lines_u = root_unlabel
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        img, target, flag = load_data(img_path, self.lines_u)
        # img = np.array(img)
        # img = torch.tensor(img).permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        # print('img shape', img.shape)
        # plt.imsave('img.png', img.numpy().transpose(1,2,0)) # 
        # plt.close()
        # assert 1 > 2
        return img, target, flag
