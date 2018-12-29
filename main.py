# -*- coding: utf-8 -*-
"""
Created on 2018/12/17 10:48

# 加载已有的model，以及对应的stage-2测试数据集，做预测prediction
# pytorch内存、GPU显存管理比较分散，一个工程中可能有多个参数涉及，熟练的话便于精细化使用。
#pytorch部分注释：
1）pin_memory：当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False
2）num_workers：表示读取样本的线程数，也就是使用多进程加载训练、测试数据的进程数，0代表不使用多进程，数量越多占用内存越大
3）collate_fn：将多个样本数据拼接成一个batch（如何拼接？），一般使用的是默认的拼接方式
4）DataBowl3Detector：是继承torch的一个子类，用于定义我们自己的数据集（加载npy、裁剪、采样、增广、拼接、label标签变化、测试阶段的切割、合并等）
5）DataLoader：在DataBowl3Detector中定义好我们自己的数据集，就可以通过torch的DataLoader来进行数据的加载
6）SplitComb.py：只在测试阶段预处理step2出现，用于imgs数据的切割与patch合并
"""

from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
import warnings

warnings.filterwarnings("ignore")

# 预测数据集的地址
datapath = config_submit['datapath']
# 预测数据集最终的预测结果地址
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    testsplit = full_prep(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
    testsplit = [f.split('_')[0] for f in os.listdir(datapath) if f.endswith('_clean.npy')]

# N-NET模型（结节检测）检测器
## detector_model.py脚本对象
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
## model初始化，并返回其他一些参数（如：config1就是anchor、训练相关的一些超参数）
config1, nod_net, loss, get_pbb = nodmodel.get_model()
## 加载训练好的weights权重，注意不是keras那样的model.load直接加载，torch.load()方法（torch.save()方法类似）
checkpoint = torch.load(config_submit['detector_param'])
## state_dict：pytorch特定的状态字典，{layer：params}这样的映射关系，而且只保存trainable_layer及对应参数
## 再把对应的网络结构和state_dict状态字典参数，做合并
nod_net.load_state_dict(checkpoint['state_dict'])

# cuda、cudnn等多GPU配置相关
torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

# bbox-cube预测结果的保存地址，因为是检测，所以会保存相关坐标
bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    # 第1阶段model的目标检测预测结果
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    # 一个batch是batch_size为1的1张原始图片，但是SplitComb在每张图片基础上提取了12个patchs，而且demo_test时是同时预测，造成GPU显存溢出
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])
    # DataBowl3Detector用来，处理预处理过后的npy数据集，提取3D的patch，并把crop维度固定为（128x128x128x1），做困难负样本挖掘
    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    # DataLoader函数把数据样本转换为pytorch特定的格式
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 2,pin_memory=False,collate_fn =collate) # 这里把num_workers调小，避免pytorch data_loader方法占用过多内存
    # 
    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

    
# C-NET模型（结节癌变分类）分类器
## classifier_model.py脚本对象
casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
## model初始化，并取第1阶段概率top=5的boxes-cube预测，作为下一阶段的输入
casenet = casemodel.CaseNet(topk=5)
## 同样有超参数
config2 = casemodel.config
## 加载训练好的weights权重，注意不是keras那样的model.load直接加载，torch.load()方法（torch.save()方法类似）
checkpoint = torch.load(config_submit['classifier_param'])
## 再把对应的网络结构和state_dict状态字典参数，做合并
casenet.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']



def test_casenet(model,testset):
    """
    第2阶段model的癌症分类预测过程，返回predlist
    :param model: 
    :param testset: 维度为（96，96，96，1），由top5的cube proposals，cropping出来的
    :return: 
    """
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 2,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        ## nodulePred：肺结节4个坐标转换系数的预测结果（原点3个坐标x、y、z与半径）
        ## casePred：肺结节预测为cancer的sigmoid概率值
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist
# 第1阶段boxes-cube输出的地址
config2['bboxpath'] = bbox_result_path
# 第2阶段cube对应的sigmoid概率与坐标转换系数的输出地址
config2['datadir'] = prep_result_path



dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
# test_casenet函数预测
predlist = test_casenet(casenet,dataset).T
# 保存结果
df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
df.to_csv(filename,index=False)
