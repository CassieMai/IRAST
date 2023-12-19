


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
from utils import densitymap_to_densitymask,densitymap_to_densitylevel
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from sklearn.metrics import r2_score


# parameter setting
dataset_path = '/maixiaochun/0_datasets/colorization/'
subset = 'experimentset'

transform = transforms.Compose([
           transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                # std=[0.229, 0.224, 0.225]),
                   ])

# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)

# with open('./part_A_test.json', 'r') as outfile:
#     img_paths = json.load(outfile)

# test images
dataset_path = os.path.join(dataset_path, subset)
test_images = os.listdir(os.path.join(dataset_path, 'test'))
test_list = []
for i in test_images:
    test_list.append(os.path.join(dataset_path, 'test', i))
img_paths = test_list


model = CSRNet()
model = model.cuda()
model.eval()

checkpoint = torch.load('0model_best.tar')
print(checkpoint['epoch'])

model.load_state_dict(checkpoint['state_dict'])


mae = 0
mse=0
count = 0
gt_count = 0
count_list = []
gt_count_list = []

for i in range(len(img_paths)):
    print(img_paths[i])
    img_path = img_paths[i]
    img = Image.open(img_path)
    img = transform(img.convert('RGB')).cuda()

    # gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images', 'ground_truth'),'r')
    # groundtruth_d = np.asarray(gt_file['density'])

    # groundtruth_d = cv2.resize(groundtruth_d, (groundtruth_d.shape[1] / 8, groundtruth_d.shape[0] / 8),
    #                            interpolation=cv2.INTER_CUBIC) * 64

    gt_path = img_path.replace('.jpg', '.csv').replace('test', 'test_den')
    gt_density = pd.read_csv(gt_path, sep=',', header=None).values
    gt_density = np.asarray(gt_density)
    h, w = gt_density.shape[0], gt_density.shape[1]
    print('gt_density', np.sum(gt_density))
    gt_density = cv2.resize(gt_density, (int(gt_density.shape[1] / 8), int(gt_density.shape[0] / 8)),
                        interpolation=cv2.INTER_NEAREST) \
                 * ((h * w) / (1.0 * int(gt_density.shape[1] / 8) * int(gt_density.shape[0] / 8)))
    print('gt_density after resize', np.sum(gt_density))
    print('img', img.unsqueeze(0).shape)
    d1, _, _, _ = model(img.unsqueeze(0))  # model(img.unsqueeze(0), 0)

    print('et:', (d1).detach().cpu().sum().numpy())
    print('gt:', np.sum(gt_density))

    mae += abs((d1).detach().cpu().sum().numpy() - np.sum(gt_density))
    mse += ((d1).detach().cpu().sum().numpy() - np.sum(gt_density))**2
    count += (d1).detach().cpu().sum().numpy()
    gt_count += np.sum(gt_density)
    print(i, mae)
    count_list.append((d1).detach().cpu().sum().numpy())
    gt_count_list.append(np.sum(gt_density))

    if False:
        print('gt_density', gt_density.shape)
        print('d1', d1.shape)
        imname = img_path.split('/')[-1][:-4]
        im = img.cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((im*255).astype('uint8')).save('visualization/'+ imname +'.png')
        plt.imsave('visualization/'+ imname +'_gt.png', gt_density, cmap='jet')
        plt.imsave('visualization/'+ imname +'_est_den.png', d1.cpu().detach().numpy()[0, :, :, :].squeeze(0), cmap='jet')

print('mae', mae/len(img_paths))
print('mse', np.sqrt(mse/len(img_paths)))
print('ratio of counting', count/(gt_count + 1e-6))
print('R2', r2_score(gt_count_list, count_list))


