#!/bin/bash

# Identical throughout the experiment
model="he_customized"
dataset="IndianPines"
#dataset="PaviaU"
trainingSample=0.8
epoch=2
cuda=0
samplingMode="fixed"
trainSet="./Datasets/${dataset}/train.mat"
testSet="./Datasets/${dataset}/test.mat"
valSet="./Datasets/${dataset}/val.mat"
bandGroup=4
lr=0.01

# Variable throughout the experiment
useKernel=(16)
useFreq=(50)
bandSelection=(64 32 16 8 4)
numOfTest=1

# With DCT
for c in ${useKernel[@]}; do 
	for f in ${useFreq[@]}; do
		for (( t=1; t<=$numOfTest; t++ )); do
			python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --band_group $bandGroup --use_freq $f --use_kernel $c --flip_augmentation --lr $lr > ./log/group_${bandGroup}_freq_${f}_ch_${c}_test_${t}.log
		done
	done
done

## Without DCT
#### Use all bands ###
#for c in ${useKernel[@]}; do
#	for (( t=1; t<=$numOfTest; t++ )); do
#		python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --flip_augmentation --lr $lr > ./log/allBands_ch_${c}_test_${t}.log
#	done
#done
#
#### Use Uniform Band Selection ### 
#selectionMode="uniform"
#for c in ${useKernel[@]}; do
#	for b in ${bandSelection[@]}; do
#		for (( t=1; t<=$numOfTest; t++ )); do
#			python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --band_selection $b --selection_mode $selectionMode --flip_augmentation --lr $lr > ./log/band_${b}_ch_${c}_${selectionMode}_test_${t}.log
#		done
#	done
#done
#
#### Use Correlation-Matrix-Based Band Selection ### 
#selectionMode="correlation"
#for c in ${useKernel[@]}; do
#	for b in ${bandSelection[@]}; do
#		for (( t=1; t<=$numOfTest; t++ )); do
#			python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --band_selection $b --selection_mode $selectionMode --flip_augmentation --lr $lr > ./log/band_${b}_ch_${c}_${selectionMode}_test_${t}.log
#		done
#	done
#done

#### Use Random Band Selection ### 
#selectionMode="random"
#for c in ${useKernel[@]}; do
#	for b in ${bandSelection[@]}; do
#		for (( t=1; t<=$numOfTest; t++ )); do
#			python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --band_selection $b --selection_mode $selectionMode --flip_augmentation --lr $lr > ./log/band_${b}_ch_${c}_${selectionMode}_test_${t}.log
#		done
#	done
#done