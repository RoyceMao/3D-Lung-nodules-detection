# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from utils import *
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc

def test_detect(data_loader, net, get_pbb, save_dir, config,n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        # target:当前索引batch的结节位置信息
        # coord:当前索引batch的所有结节位置信息
        # nzhw:
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False # isfeat=False不变
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = n_gpu
        # data = data[0:1] # 每张图片的所有12个patchs中只取1个patch做预测，后续会有合并上的问题
        print(data.size()) # (12,1,208,208,208)，原图batch_size为1，因为patch裁剪和采样，有12个patchs被加载进来。相当于同时预测12张图片，导致GPU显存溢出
        splitlist = range(0,len(data)+1,n_gpu)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                # RuntimeError: DataLoader worker (pid 22623) is killed by signal: Killed.(内存溢出的问题，调低num_workers和样本图片数量可解决)
                # RuntimeError: CUDA out of memory. Tried to allocate xxx.xx MiB.(GPU显存溢出的问题，减少同时做预测的patchs数量，或者加入del output)
                output = net(input,inputcoord) 
            outputlist.append(output.data.cpu().numpy())
            del output # 避免GPU显存溢出加上该句
        output = np.concatenate(outputlist,0)
        # combine操作结合所有的patch计算出最终的预测结果
        output = split_comber.combine(output,nzhw=nzhw) 
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]
        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,shortname])
        e = time.time()
        
        np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print
