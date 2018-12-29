# -*- coding: utf-8 -*-
#!/bin/bash
# GPU1080ti 数量(1块)
#受限于机器配置，交替训练第1阶段n-net的batch_size不超过4
#受限于机器配置，交替训练第2阶段c-net的batch_size不超过2、1
set -e

python prepare.py
cd detector
eps=100
CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 4 --epochs $eps --save-dir ./results/res18 
# CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 4 --resume results/res18/$eps.ckpt --test 0 # 测试输出，用于第2阶段的输入（这里的main脚本未做修改，只修改了3D-Lung-nodules-detection根目录下的main.py）
cp results/res18/$eps.ckpt ../../model/detector.ckpt

cd ../classifier
# python adapt_ckpt.py --model1  net_detector_3 --model2  net_classifier_3  --resume ../detector/results/res18/$eps.ckpt # 设立初始的start.ckpt
CUDA_VISIBLE_DEVICES=0 python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 2 -b2 1 --save-dir net3 --resume ./results/start.ckpt --start-epoch 30 --epochs 130
CUDA_VISIBLE_DEVICES=0 python main.py --model1  net_detector_3 --model2  net_classifier_4 -b 2 -b2 1 --save-dir net4 --resume ./results/net3/130.ckpt --freeze_batchnorm 1 --start-epoch 121
cp results/net4/160.ckpt ../../model/classifier.ckpt
