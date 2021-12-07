#!/usr/bin/env bash
set -x
DATAPATH="/home/jarvis/Research/dl_projects/datasets/KITTI/kitti_2015"
CUDA_VISIBLE_DEVICES=0 python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet --maxdisp 256 \
--loadckpt "pretrained_models/sceneflow_pretraining.ckpt"
# --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/robust_pretrain55/300_100_final50/best_upload"
