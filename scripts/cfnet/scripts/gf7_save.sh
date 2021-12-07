#!/usr/bin/env bash
set -x
DATAPATH="/home/jarvis/Research/dl_projects/datasets/gf7_subset"
SAVEPATH="./results/gf7_test"
CUDA_VISIBLE_DEVICES=0 python save_disp_gf7.py --dataset custom --datapath $DATAPATH --savepath $SAVEPATH --testlist ./filenames/whu_stereo_test.txt --model cfnet --maxdisp 256 \
--loadckpt "pretrained_models/sceneflow_pretraining.ckpt"
# --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/robust_pretrain55/300_100_final50/best_upload"
