#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
experiment=1
run_name='Experiment'
rounds=2 epochs=150 patience=10 device='mps'
batch_size=1024
record=1 program_test=0 verbose=1 classification=0
dimensions="30"
datasets="cpu"
densities="0.80"
py_files="train_model"
#models="neucf"
models="mf neucf"

for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
            for density in $densities
            do
                for model in $models
                do
										python ./$py_file.py \
													--device $device \
													--logger $run_name \
													--rounds $rounds \
													--density $density \
													--dataset $dataset \
													--patience $patience \
													--model $model \
													--bs $batch_size \
													--epochs $epochs \
													--patience $patience \
													--bs $batch_size \
													--program_test $program_test \
													--dimension $dim \
													--experiment $experiment \
													--record $record \
													--verbose $verbose \
													--classification $classification

                done
            done
        done
    done
done