import random
import os
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance,ImageOps,ImageChops
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import pandas as pd


def load_data(img_path, unlabel_image_path):
    gt_path = None
    if 'train' in img_path:
        gt_path = img_path.replace('.png', '.csv').replace('train', 'train_den')
    elif 'test' in img_path:
        gt_path = img_path.replace('.png', '.csv').replace('test', 'test_den')
    else:
        assert 'train' in img_path or 'test' in img_path
    img = Image.open(img_path).convert('RGB')
    gt_density = pd.read_csv(gt_path, sep=',', header=None).values

    if img_path in unlabel_image_path:
        # for unlabeled images, density maps are unavailable
        flag = 1
        target = np.asarray(0)
        # print('target', target)
    else:
        flag=0
        target = np.asarray(gt_density)
        h, w = target.shape[0], target.shape[1]
        # print('target', np.sum(target))
        target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0]/ 8)),
                                   interpolation=cv2.INTER_NEAREST) * ((h*w)/(1.0*int(target.shape[1] / 8)*int(target.shape[0]/ 8)))
        # print('target after resize', np.sum(target))

    return img, target, flag
