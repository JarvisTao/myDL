from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--savepath', default='./results/tmp', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
print(test_dataset.left_filenames[:10])
test_dataset.load_part_img(suffix = '068_')
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

# check savepath
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

def test():
    #os.makedirs('./predictions', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        left_img = sample["left_ori"].squeeze()
        right_img = sample["right_ori"].squeeze()
        # left_img = tensor2numpy(sample["left"].squeeze().permute(1,2,0))*255
        # left_img = left_img.astype(np.uint8)
        # right_img = tensor2numpy(sample["right"].squeeze().permute(1,2,0))*255
        # right_img = right_img.astype(np.uint8)
        disp_est_np = tensor2numpy(test_sample(sample, reverse=False))
        disp_gt = tensor2numpy(sample["disparity"].squeeze())
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            if args.dataset == 'igarss' or args.dataset == 'custom':
                disp_est = np.array(disp_est, dtype=np.float32)
                disp_gt = np.array(disp_gt, dtype=np.float32)
                disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
                disp_gt = np.array(disp_gt[top_pad:, :-right_pad], dtype=np.float32)
                # disp_est[disp_est > 2 * (disp_gt.max())] = 0.0
                disp_gt[disp_gt == -999.0] = 0.0
            else:
                disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
                disp_gt = np.array(disp_gt[top_pad:, :-right_pad], dtype=np.float32)
            #fn = os.path.join("predictions", fn.split('/')[-1])
            # fn = os.path.join("/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/pre_picture/", fn.split('/')[-1])
            # fn = os.path.join(args.savepath, fn.split('/')[-1])
            fn = os.path.join(args.savepath, fn.replace('/','_'))
            print("saving to", fn, disp_est.shape)
            # disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            print('disp_est:',disp_est.max(), disp_est.min())
            print('disp_gt:',disp_gt.max(), disp_gt.min())
            # print(disp_est_uint.max(), disp_est_uint.min())
            # im2show = cv2.applyColorMap(disp_est_uint, 2)
            plt.figure(figsize=(20, 5.5))
            # plt.figure(figsize=(16, 8))
            plt.subplot(141)
            plt.imshow(left_img)
            plt.axis('off')
            plt.subplot(142)
            plt.imshow(right_img)
            plt.axis('off')
            ax = plt.subplot(143)
            im = plt.imshow(disp_est, 'gray')
            ax.axis('off')
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, shrink=0.9, cax=cax)
            ax = plt.subplot(144)
            im = plt.imshow(disp_gt, 'gray')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, shrink=0.9, cax=cax)
            ax.axis('off')
            # plt.get_current_fig_manager().window.showMaximized()
            plt.savefig(fn)
            # plt.close()
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.08, hspace=0)
            plt.show()
            # skimage.io.imsave(fn, disp_est_uint)


# test one sample
@make_nograd_func
def test_sample(sample, reverse = False):
    model.eval()
    if reverse:
        disp_ests, pred1_s3_up, pred2_s4 = model(sample['right'].cuda(), sample['left'].cuda())
        print('reverse: ', reverse)
    else:
        disp_ests, pred1_s3_up, pred2_s4 = model(sample['left'].cuda(), sample['right'].cuda())
        print('reverse:', reverse)
    return disp_ests[-1]


if __name__ == '__main__':
    test()
