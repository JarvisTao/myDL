from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
from datasets import __datasets__
from datasets.igarss_dataset import IgarssDatasetSimple
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='igarss', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/jarvis/Research/datasets/igrass', help='data path')
parser.add_argument('--testlist', default='./filenames/igarss_train.txt', help='testing list')

# parse arguments
args = parser.parse_args()

print(args.datapath)
# dataset, dataloader
# StereoDataset = __datasets__[args.dataset]
# test_dataset = StereoDataset(args.datapath, args.testlist, False)
test_dataset = IgarssDatasetSimple(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
# TestImgLoader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4, drop_last=False)

print(len(test_dataset))
cnt = len(test_dataset) * 2

data = next(iter(TestImgLoader))

print(data['left'].shape)
print(data['left'].mean())

mean = torch.zeros(3)
std  = torch.zeros(3)

for batch in tqdm(TestImgLoader):
    for d in range(3):
        mean[d] += batch['left'][:,d,:,:].mean()
        mean[d] += batch['right'][:,d,:,:].mean()
        std[d] += batch['left'][:,d,:,:].std()
        std[d] += batch['right'][:,d,:,:].std()

mean.div_(cnt)
std.div_(cnt)
print(mean,std)
