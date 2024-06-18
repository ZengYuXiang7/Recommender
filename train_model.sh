#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
python train_model.py --config_path ./exper_config.py --exp_name MFConfig --experiment 1
#python train_model.py --config_path ./exper_config.py --exp_name CFConfig --experiment 1
#python train_model.py --config_path ./exper_config.py --exp_name NeuCFCnfig --experiment 1