# -*- coding: utf-8 -*-
"""
Created on 2018/12/17 16:09

# stage1训练数据集以及luna原始数据集的预处理阶段
"""
import os
import shutil
import numpy as np
from config_training import config


from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('../preprocessing')
from step1 import step1_python
import warnings
# np.set_printoptions(threshold=np.inf) # 允许numpy数组的完全打印

def resample(imgs, spacing, new_spacing,order=2):
    """
    重采样，即采用新分辨率
    :param imgs: 
    :param spacing: 
    :param new_spacing: 
    :param order: 
    :return: 
    """
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        # zoom 将图片每一维以相应系数缩小
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    世界坐标转换为真实坐标
    :param worldCoord: 
    :param origin: 
    :param spacing: 
    :return: 
    """
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    """
    itk是专业的医学图像处理库，用来加载3D维度的CT、MRC等图像
    :param filename: 
    :return: 
    """
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename) # 读取.mhd文件
    numpyImage = sitk.GetArrayFromImage(itkimage) # 获取数据，自动从同名的.raw文件读取
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin()))) # 原始CT坐标系的坐标原点coords
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing()))) # 原始CT坐标系的体素距离
    # 返回CT坐标系下的标注信息
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    """
    mask标注预处理相关
    :param mask: 
    :return: 
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        # np.ascontiguousarray方法copy返回一个跟参数array有一样shape的连续数组
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0: # 有mask目标
            # convex_hull_image是CV高级形态学处理函数，输入为二值图像，输出一个逻辑二值图像，在凸包内的点为True, 否则为False
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1): # 凸包生成的凸多边形太过分，掩盖了原始mask1的大致形状信息，则放弃凸包处理
                mask2 = mask1
        else: # 没有mask目标
            mask2 = mask1
        convex_mask[i_layer] = mask2
    # 二值膨胀
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)
    # 返回膨胀后的mask
    return dilatedMask


def lumTrans(img):
    """
    灰度标准化（HU值），将HU值（[-1200, 600]）线性变换至0~255内的灰度值
    :param img: 
    :return: 
    """
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def savenpy(id,annos,filelist,data_path,prep_folder):
    """
    图像预处理重点-提取肺部区域ROI，以及提取的数据保存为numpy
    :param id: 
    :param annos: 
    :param filelist: 
    :param data_path: 
    :param prep_folder: 
    :return: 
    """
    resolution = np.array([1,1,1])
    name = filelist[id]
    label = annos[annos[:,0]==name]
    # z轴坐标放在第一位
    label = label[:,[3,1,2,4]].astype('float')
    # 切片后，im是像素点HU值，m1是只有目标标注的mask像素值，m2是只有非目标区域标的的mask像素值，spacing是体素距离
    im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
    Mask = m1+m2
    # mask世界坐标系到真实坐标系的转换
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    # box立方体的坐标表示形式
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    # np.expand_dims：用于扩展数组的形状，spacing.shape=（3，2），resolution.shape=（3，2）
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    # np.floor：返回不大于输入参数的最大整数
    box = np.floor(box).astype('int')
    # 每个轴左右方向上的扩张像素值
    margin = 5
    # box向外扩张10个像素
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    # 膨胀后的mask
    dilatedMask = dm1+dm2
    # 原始的mask
    Mask = m1+m2
    # 膨胀新增区域
    extramask = dilatedMask - Mask
    # 像素灰度值阈值（骨头区域）
    bone_thresh = 210
    # mask以外区域的像素值
    pad_value = 170
    # HU值初始化为（-1000，1000）？
    im[np.isnan(im)]=-2000
    # HU值（[-1200, 600]）线性变换至0~255内的灰度值
    sliceim = lumTrans(im)
    # 切片img中mask以外的像素灰度值均设为170
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    # 切片img中大于像素灰度值阈值的像素点（骨头区域）
    bones = sliceim*extramask>bone_thresh
    # 切片img中像素灰度值高于210的区域灰度值也设为170
    sliceim[bones] = pad_value
    # 像素距离标准化（体素值进行映射与归一化）
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    # 左右肺的凸包扩张处理
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    # name+'_clean.npy'：就是去除肺部周边噪声后的，3D切片图数组
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    # name+'_label.npy'：2D切片图相应的label数组
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)

    print(name)

def full_prep(step1=True,step2 = True):
    """
    同时操作多个文件目录（data_path下生成的filelist），实现多进程multiprocessing，处理的预处理函数就是savenpy
    :param step1: 
    :param step2: 
    :return: 
    """
    warnings.filterwarnings("ignore")

    #preprocess_result_path = './prep_result'
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    finished_flag = '.flag_prepkaggle'
    
    if not os.path.exists(finished_flag):
        alllabelfiles = config['stage1_annos_path']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:,0]!=np.nan]
            tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)
        filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        print('starting preprocessing')
        pool = Pool()
        filelist = [f for f in os.listdir(data_path)]
        # 高阶嵌套函数的调用，用于固定1个或多个初始值，返回的是一个可调用的partial对象，partial(func,*args,**kw)
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    f= open(finished_flag,"w+")        

def savenpy_luna(id,annos,filelist,luna_segment,luna_data,savepath):
    """
    与kaggle stage数据集的原理没什么差异
    :param id: 
    :param annos: 
    :param filelist: 
    :param luna_segment: 
    :param luna_data: 
    :param savepath: 
    :return: 
    """
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int') # 获取mask在新分辨率下的尺寸
    m1 = Mask==3 # LUNA16的掩码有两种值3和4？？？
    m2 = Mask==4
    Mask = m1+m2 # 将两种掩码合并
    
    xx,yy,zz= np.where(Mask) # 确定掩码的边界
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) # 对边界即掩码的最小外部长方体应用新分辨率
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    this_annos = np.copy(annos[annos[:,0]==str(name)]) # 读取该病例对应标签

    if isClean: # luna的数据集噪声较少，如果需要处理部分噪声
        convex_mask = m1
        dm1 = process_mask(m1) # 对掩码采取膨胀操作，去除肺部黑洞
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim) # 对原始数据阈值化，并归一化
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1) # 对原始数据重采样，即采用新分辨率
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1], # 将extendbox内数据取出作为最后结果
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath,name+'_clean.npy'),sliceim)


    if islabel: # label标签的numpy
        this_annos = np.copy(annos[annos[:,0]==str(name)]) # 一行代表一个结节，一个病例可能对应多行标签
        label = []
        if len(this_annos)>0:
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing) # 将世界坐标转换为真实的体素坐标
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]])) # 这1段dataframe的操作没看懂？？？
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]]) # #若没有结节则设为全0，第1次读取label_numpy就是全设为0的问题
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) # 对标签应用新的分辨率
            label2[3] = label2[3]*spacing[1]/resolution[1] # 对直径应用新的分辨率
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1) # 将box外的长度砍掉，也就是相对于box的坐标
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'),label2)
        
    print(name)

def preprocess_luna():
    """
    同时操作多个文件目录（data_path下生成的filelist），实现多进程multiprocessing，处理的预处理函数就是savenpy_luna
    :return: 
    """
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag): # 先不管结束标志对代码的运行影响
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # 开启线程池
        pool = Pool()
        #函数修饰器，将一些参数预先设定，后面调用更简洁
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                       luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist) # N=89
        #savenpy(1)
        #将函数调用在序列的每个元素上，返回一个含有所有返回值的列表
        _=pool.map(partial_savenpy_luna,range(N)) # 开多线程池可能会出现内存不够用MemoryError，这时可以接着上次的index来
        pool.close()# 关闭线程池
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")# 预处理结束，写入结束标志
    ### 最终处理生成了81张训练集图片，还有遗漏的7张暂不影响训练测试
def prepare_luna():
    """
    这部分因为luna16已经帮我们处理好了，所以不用再prepare
    :return: 
    """
    print('start changing luna name')
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):

        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

#         allnames = []
#         for d in subsetdirs:
#             files = os.listdir(d)
#             names = [f[:-4] for f in files if f.endswith('mhd')]
#             allnames = allnames + names
#         allnames = np.array(allnames)
#         allnames = np.sort(allnames)

#         ids = np.arange(len(allnames)).astype('str')
#         ids = np.array(['0'*(3-len(n))+n for n in ids])
#         pds = pandas.DataFrame(np.array([ids,allnames]).T)
#         namelist = list(allnames)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]
        
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))
                print(os.path.join(luna_data,str(id)+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                print(os.path.join(luna_segment,filename+lastfix))


        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")
    
if __name__=='__main__':
    # full_prep(step1=True,step2=True)
    # prepare_luna()
    preprocess_luna()
    
