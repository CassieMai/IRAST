import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint0, \
    densitymap_to_densitylevel, \
    densitymap_to_densitymask, unlabel_CE_loss4v1, \
    unlabel_CE_loss3v1, unlabel_CE_loss2v1  # generate_th_for_class

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
import random
import dataset
import time
import math

import scipy.io as io
import glob
import PIL.Image as Image
import h5py
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN', default='./part_A_train_with_val.json',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST', default='./part_A_val.json',
                    help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default='', type=str,
                    help='path to the pretrained model')
parser.add_argument('--gpu', metavar='GPU', type=str, default='0', help='GPU id to use.')
parser.add_argument('--task', metavar='TASK', type=str, default='debug', help='task id to use.') # ======

parser.add_argument('--dataset_path', type=str, default='/home/xiaocmai/scratch/datasets/colorization/')
parser.add_argument('--subset', type=str, default='debugset', choices=['debugset', 'experimentset'])


def main():
    
    global args, best_prec1
    
    best_prec1 = 1e6
    best_prec2 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.batch_size = 1
    args.momentum = 0.9
    args.decay = 5*1e-4
    args.start_epoch = 0
    args.epochs = 100   # ========
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 110  # ========

    args.channel = 10
    args.th = 0.9
    args.max_epochs = 120  # =======
    args.max_val = 0.1
    args.max_val1 = 0.1
    args.ramp_up_mult = -5
    args.k = 30
    args.n_samples = 270  # ======
    args.alpha = 0.7
    args.global_step = 0
    args.Z = 300 * ['']  ##intermediate outputs
    args.z = 300 * ['']  ##temporal outputs
    args.epsilor = 4e-2
    args.T = 0.5
    args.gap = 0.1
    args.beta = 1e-3
    args.max_update_pixel = None

    # with open(args.train_json, 'r') as outfile:
        # train_list = json.load(outfile)
        # using it for part a
        # train_list_unlabel = train_list[0:81] + train_list[93:173] + train_list[184:214] + train_list[218:235] + train_list[237:239]
        # train_list_label = train_list[81:93] + train_list[173:184] + train_list[214:218] + train_list[235:237]+ train_list[239:240]

    # with open(args.test_json, 'r') as outfile:
    #     val_list = json.load(outfile)
    #     val_list_unlabel = []
    #     val_list_label = val_list

    # # Orange dataset
    dataset_path = os.path.join(args.dataset_path, args.subset)
    train_images = os.listdir(os.path.join(dataset_path, 'train'))
    val_images = os.listdir(os.path.join(dataset_path, 'test'))
    train_list = []
    val_list = []
    for i in train_images:
        train_list.append(os.path.join(dataset_path, 'train', i))
    for i in val_images:
        val_list.append(os.path.join(dataset_path, 'test', i))

    train_labeled_num = int(len(train_list) * 0.4)  # TODO: revise
    train_list_unlabel = train_list[train_labeled_num:]
    train_list_label = train_list[:train_labeled_num]  # TODO: revise to load labeled image names in txt file
    val_list_unlabel = []
    val_list_label = val_list

    print('trainlist\n', len(train_list))
    print('train_list_unlabel\n', len(train_list_unlabel))
    print('train_list_label\n', len(train_list_label))
    print('val_list_label\n', len(val_list_label))


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    rand_seed = 123456
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    random.seed(rand_seed)
    
    model = CSRNet()
    model = model.cuda()

    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_cls = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=10).cuda() # if 10, the pixel is invalid

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        train(train_list_unlabel, train_list_label, model, criterion_mse, criterion_cls, optimizer, epoch)

        prec1 = validate(val_list_unlabel, val_list_label, model)

        index1 = epoch
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        if is_best:
            best_index = index1
        print ('best_mae is', best_prec1)
        save_checkpoint0({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'index': index1
            }, is_best, args.task)


def train(train_list_unlabel, train_list_label, model, criterion_mse, criterion_cls, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list_unlabel, train_list_label,
                            shuffle = True,
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    #  std=[0.229, 0.224, 0.225]),  # ==== TODO: revise
                            ]),
                            train = True,
                            seen = model.seen,
                            batch_size = args.batch_size,
                            num_workers = args.workers,
                            ),
        batch_size = args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target, flag) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = Variable(img.float().cuda())

        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        target = F.relu(target)

        density, logits2, logits3, logits4 = model(img)

        if flag.sum() == 0: 
            # processing labeled images
            target_mask_2 = densitymap_to_densitymask(target, threshold=0.0)
            target_mask_3 = densitymap_to_densitymask(target, threshold=0.048)
            target_mask_4 = densitymap_to_densitymask(target, threshold=0.258)

            loss_density_1 = criterion_mse(density, target)

            loss_cls_label2 = criterion_cls(logits2, target_mask_2)
            loss_cls_label3 = criterion_cls(logits3, target_mask_3)
            loss_cls_label4 = criterion_cls(logits4, target_mask_4)

            loss = 1 * loss_density_1 + 0.01 * (loss_cls_label2.sum() + loss_cls_label3.sum() + loss_cls_label4.sum())/3.0

        else: 
            # processing unlabeled images
            pro2u, pro3u, pro4u = F.softmax(logits2, dim=1), F.softmax(logits3, dim=1), F.softmax(logits4, dim=1)

            loss_cls_unlabel2 = unlabel_CE_loss2v1(logits2=logits2, prob3=pro3u, prob4=pro4u, th=args.th,
                                                max_update_pixel=args.max_update_pixel, criterion_cls=criterion_cls)
            loss_cls_unlabel3 = unlabel_CE_loss3v1(prob2=pro2u, logits3=logits3, prob4=pro4u, th=args.th,
                                                max_update_pixel=args.max_update_pixel, criterion_cls=criterion_cls)
            loss_cls_unlabel4 = unlabel_CE_loss4v1(prob2=pro2u, prob3=pro3u, logits4=logits4, th=args.th,
                                                max_update_pixel=args.max_update_pixel, criterion_cls=criterion_cls)
                     
            loss = 0.01 * (loss_cls_unlabel2.sum() + loss_cls_unlabel3.sum() + loss_cls_unlabel4.sum())/3.0

        losses.update(loss, img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_list_unlabel, val_list_label, model):
    print ('begin test ------------------')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list_unlabel, val_list_label,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #                      std=[0.229, 0.224, 0.225]),
                            ]), train=False
                            ),
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target, flag) in enumerate(test_loader):
        img = img.cuda()

        img = Variable(img.float())
        target = target.type(torch.FloatTensor).unsqueeze(0)
        target = target.cuda()
        target = Variable(target)  # *mask_roi_v sharp

        d1, _, _, _ = model(img)
        d = d1

        mae += abs((d.data.sum() - target.sum())) / (target.sum() + 1e-6)

    mae_min = mae / len(test_loader)
    
    print(' * MAE {mae:.3f} '.format(mae=mae_min))

    return mae_min


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


if __name__ == '__main__':
    main()        
