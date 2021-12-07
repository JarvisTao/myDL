#!/usr/bin/env bash
set -x
DATAPATH="/home/jarvis/Research/dl_projects/datasets/WHU_stereo_dataset"
SAVEPATH="./results/whu_test"
CUDA_VISIBLE_DEVICES=0 python save_disp.py --dataset whu --datapath $DATAPATH --savepath $SAVEPATH --testlist ./filenames/whu_stereo_test.txt --model cfnet --maxdisp 256 \
--loadckpt "pretrained_models/sceneflow_pretraining.ckpt"
# --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/robust_pretrain55/300_100_final50/best_upload"
