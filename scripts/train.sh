#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train_drum.py \
    --size 64 --batch 8 --sample_dir sample_Bandlab_Beats_one_bar \
    --checkpoint_dir checkpoint_Bandlab_Beats_one_bar \
    "D:\LoopTest\data\processed\mel_80_320"
