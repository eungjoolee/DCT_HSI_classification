#!/bin/bash

# Identical throughout the experiment
model="he_customized"
dataset="IndianPines"
trainingSample=0.8
epoch=100
cuda=0
samplingMode="fixed"
trainSet="./Datasets/${dataset}/train.mat"
testSet="./Datasets/${dataset}/test.mat"
valSet="./Datasets/${dataset}/val.mat"
bandGroup=4
lr=0.01

## Variable throughout the experiment
#useKernel=(16)
#useFreq=(50 16 8 4 2 1)
#numOfTest=1

## With DCT
#for c in ${useKernel[@]}; do 
#	for f in ${useFreq[@]}; do
#		for (( t=1; t<=$numOfTest; t++ )); do
#			python main.py --model $model --dataset $dataset --training_sample $trainingSample \
#				--epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet \
#				--band_group $bandGroup --use_freq $f --use_kernel $c --flip_augmentation --lr $lr > ./log/group_${bandGroup}_freq_${f}_ch_${c}_test_${t}.log
#		done
#	done
#done

t_bandGroup=4
t_useKernel=16
t_useFreq=50
t_restore="./checkpoints/he_et_al_customized/IndianPines/teacher/2021-04-17_02:09:25.519213_epoch100_0.90_gp_4_fq_50_ch_16.pth"
kd_alpha=0.9
kd_temp=20

s_bandGroup=4
s_useKernel=16
s_useFreq=2

# KD
python main.py --model $model --dataset $dataset --training_sample $trainingSample \
		--epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet \
		--band_group $s_bandGroup --use_freq $s_useFreq --use_kernel $s_useKernel --flip_augmentation --lr $lr \
		--t_band_group $t_bandGroup --t_use_kernel $t_useKernel --t_use_freq $t_useFreq \
		--t_restore $t_restore --kd_alpha $kd_alpha --kd_temp $kd_temp > ./log/kd_freq_${t_useFreq}_ch_${t_useKernel}_to_freq_${s_useFreq}_ch_${s_useKernel}.log