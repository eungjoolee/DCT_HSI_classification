#!/bin/bash

##############  4 groups  ##############

############## 16 kernels ##############
# w/o DCT (200 bands)
python main.py --model he_full --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --use_kernel 16 > ./log/band_200_CH_16.log

# 50 frequencies
python main.py --model he_full --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 50 --use_kernel 16 > ./log/group_4_freq_50_CH_16.log

# 16 frequencies
python main.py --model he_full --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 16 --use_kernel 16 > ./log/group_4_freq_16_CH_16.log

# 8 frequencies (he_32)
python main.py --model he_32 --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 8 --use_kernel 16 > ./log/group_4_freq_8_CH_16.log

# 4 frequencies (he_16)
python main.py --model he_16 --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 4 --use_kernel 16 > ./log/group_4_freq_4_CH_16.log

# 2 frequencies (he_8)
python main.py --model he_8 --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 2 --use_kernel 16 > ./log/group_4_freq_2_CH_16.log

# 1 frequencies (he_4)
python main.py --model he_4 --dataset IndianPines --training_sample 0.8 --epoch 200 --cuda 0 --sampling_mode fixed --train_set train.mat --test_set test.mat --val_set val.mat --band_group 4 --use_freq 1 --use_kernel 16 > ./log/group_4_freq_1_CH_16.log