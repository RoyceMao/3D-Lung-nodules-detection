# 3D-Lung-nodules-detection
<br><br>
## 环境
   * Python 2.7.15
   * pytorch 1.0.0
   
## 检测效果（结节预测3D-Cube的2D切片展示）
<img src="https://github.com/RoyceMao/3D-Lung-nodules-detection/blob/master/img/EG.png" width="402" height="299"/>

## 数据集
   [LUNA16](https://luna16.grand-challenge.org/)
   
## GPU版Pytorch安装
```
//注意对应的cuda版本
conda install pytorch torchvision cuda90 -c pytorch
```
## Dicom格式的CT影像转(raw,mhd)格式(已经是raw+mhd就不用转了)
```
//raw就是同一病例的dicom影像在z轴上的堆叠，它包含了一个病例所有切片的原始数据
python dicom2raw.py
```
<img src="https://github.com/RoyceMao/3D-Lung-nodules-detection/blob/master/img/EG1.png" width="450" height="200"/>

## labels提取
因为candidates.csv中是针对所有切片的结节class标注，不是单个病例的class，模型第2阶段是在单个病例基础上做癌症分类的判断。
```
//train文件中，用于train与validation的图片数量之比为4：1，test文件中由于没有对应的掩模标注，只能拿来做预测
python labels_extract.py
```

## numpy数据预处理
预处理分为step1、step2两阶段，step1的过程（主要去噪、坐标转换、统一分辨率、提取ROI、提取标签numpy），只是最终输出的不是切片，还是prepare的中间3D图片结果（各图片size不一），step2包含3D-patch裁剪、困难负样本挖掘、数据增广、拼接、以及测试阶段的切割、合并等，需要注意的是：训练和测试的各自输入patch size大小是不一样的，<br>
    训练：（None,1,128,128,128）<br>
    测试：（None,1,208,208,208）<br>
```
// step1的预处理过程，于./work/preprocess目录下生成name_clean.npy、name_label.npy
python prepare.py
// step2的预处理直接封装在了DataBowl3Detector类中
```

## 训练预测
输入stage1_numpy：name_clean.npy、name_label.npy<br>
输入stage2_numpy：name_pbb.npy、name_lbb.npy<br>

```
//2阶段交替式**训练**（注：shell脚本中注释掉的1阶段输出，用根目录下的main.py脚本代替）
bash run_training.sh
//cd于工程根目录做**预测**
python main.py
```

## 具体日志及预测效果见jupyter、csv文件
N-NET
```
using gpu 0
Epoch 001 (lr 0.01000)
Train:      tpr 15.15, tnr 21.81, total pos 66, total neg 188, time 42.76
loss 0.9499, classify loss 0.8090, regress loss 0.0326, 0.0225, 0.0410, 0.0447
 
Epoch 002 (lr 0.01000)
Train:      tpr 4.55, tnr 34.04, total pos 66, total neg 188, time 28.38
loss 0.7921, classify loss 0.7195, regress loss 0.0111, 0.0104, 0.0099, 0.0412
 
Epoch 003 (lr 0.01000)
Train:      tpr 7.58, tnr 38.30, total pos 66, total neg 188, time 28.11
loss 0.7659, classify loss 0.7085, regress loss 0.0093, 0.0091, 0.0091, 0.0299
...
```
C-NET
```
Train, epoch 30, loss2 1.2462, miss loss 0.0000, acc 0.6000, tpn 3, fpn 2, fnn 0, time 6.88, lr  0.00000
Epoch 030 (lr 0.01000)
Train:      tpr 39.39, tnr 68.09, total pos 66, total neg 188, time 39.00
loss 0.8102, classify loss 0.7199, regress loss 0.0098, 0.0085, 0.0119, 0.0601
 
Train, epoch 30, loss2 0.7002, miss loss 0.0863, acc 0.7037, tpn 55, fpn 20, fnn 4, time 49.96, lr  0.01000
Epoch 031 (lr 0.01000)
Train:      tpr 4.55, tnr 97.34, total pos 66, total neg 188, time 33.07
loss 0.7508, classify loss 0.6883, regress loss 0.0083, 0.0082, 0.0115, 0.0345
 
Train, epoch 31, loss2 0.5906, miss loss 0.0000, acc 0.7037, tpn 54, fpn 19, fnn 5, time 50.17, lr  0.01000
Epoch 032 (lr 0.01000)
Train:      tpr 1.52, tnr 97.34, total pos 66, total neg 188, time 33.07
loss 0.7061, classify loss 0.6507, regress loss 0.0068, 0.0073, 0.0076, 0.0337
...
```