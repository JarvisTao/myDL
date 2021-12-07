#!/usr/bin/env bash
set -x
# DATAPATH="/home/jarvis/Research/dl_projects/datasets/KITTI/"
DATAPATH="/home/jarvis/Research/dl_projects/datasets/WHU_stereo_dataset"
CUDA_VISIBLE_DEVICES=0 python robust_test.py --dataset whu \
    --datapath $DATAPATH --trainlist ./filenames/whu_stereo_train.txt --batch_size 4 --test_batch_size 2 \
    --testlist ./filenames/whu_stereo_test.txt --maxdisp 256 \
    --epochs 1 --lr 0.001  --lrepochs "300:10" \
		--loadckpt "pretrained_models/sceneflow_pretraining.ckpt" \
    --model cfnet --logdir ./checkpoints/robust_abstudy_test

    # --datapath $DATAPATH --trainlist ./filenames/kitti15_errortest.txt --batch_size 4 --test_batch_size 2 \
    # --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/sceneflow_doubletrain/mish45_55/checkpoint_000032.ckpt" \
