#!/usr/bin/env bash
set -x
# DATAPATH="/home1/datasets/Database/robust/kitti12_15/"
DATAPATH="/home/jarvis/Research/datasets/igrass/"
python -m torch.distributed.launch --nproc_per_node 2 robust.py --dataset igarss \
    --datapath $DATAPATH --trainlist ./filenames/igarss_train.txt --batch_size 4 --test_batch_size 1 \
    --testlist ./filenames/igarss_test.txt --maxdisp 256 \
    --epochs 400 --lr 0.001  --lrepochs "300:10" \
		--loadckpt "pretrained_models/sceneflow_pretraining.ckpt" \
    --model cfnet --logdir ./checkpoints/robust_abstudy_test_igarss \
		--save_freq 20 

		# --resume \
# CUDA_VISIBLE_DEVICES=0 python robust.py --dataset kitti \
		# --loadckpt "checkpoints/robust_abstudy_test/checkpoint_000244.ckpt" \
		#--loadckpt "pretrained_models/sceneflow_pretraining.ckpt" \
    # --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/sceneflow_doubletrain/mish45_55/checkpoint_000032.ckpt" \
