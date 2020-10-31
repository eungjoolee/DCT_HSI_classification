#!/bin/bash

# Identical throughout the experiment
model="he_customized"
dataset="IndianPines"
trainingSample=0.8
epoch=200
cuda=0
samplingMode="fixed"
trainSet="train.mat"
testSet="test.mat"
valSet="val.mat"
bandGroup=4

# Variable throughout the experiment
useKernel=(16 8 4)
useFreq=(50 16 8 4 2 1)
bandSelection=(64 32 16 8 4)
numOfTest=5

# Without DCT
### Use all bands ###
for c in ${useKernel[@]}; do
	python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c > ./log/allBands_ch_${c}.log
done

### Use Random Band Selection ### 
selectionMode="random"
for c in ${useKernel[@]}; do
	for b in ${bandSelection[@]}; do
		for (( t=1; t<=$numOfTest; t++ )); do
			python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --band_selection $b --selection_mode $selectionMode > ./log/band_${b}_ch_${c}_${selectionMode}_test_${t}.log
		done
	done
done

### Use Uniform Band Selection ### 
selectionMode="uniform"
for c in ${useKernel[@]}; do
	for b in ${bandSelection[@]}; do
		python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --use_kernel $c --band_selection $b --selection_mode $selectionMode > ./log/band_${b}_ch_${c}_${selectionMode}.log
	done
done

# With DCT
for c in ${useKernel[@]}; do 
	for f in ${useFreq[@]}; do
		python main.py --model $model --dataset $dataset --training_sample $trainingSample --epoch $epoch --cuda $cuda --sampling_mode $samplingMode --train_set $trainSet --test_set $testSet --val_set $valSet --band_group $bandGroup --use_freq $f --use_kernel $c > ./log/group_${bandGroup}_freq_${f}_ch_${c}.log
	done
done