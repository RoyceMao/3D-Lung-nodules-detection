# -*- coding: utf-8 -*-
config = {'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
          'luna_raw':'/work/subset0',
          'luna_segment':'../work/seg-lungs-LUNA16/',
          
          'luna_data':'../work/subset0',
          'preprocess_result_path':'../work/preprocess',
          'preprocess_path':'../../work/preprocess',
          
          'luna_abbr':'./detector/labels/shorter.csv',
          'luna_label':'../work/annotations.csv',
          'stage1_annos_path':['./detector/labels/label_job5.csv',
                './detector/labels/label_job4_2.csv',
                './detector/labels/label_job4_1.csv',
                './detector/labels/label_job0.csv',
                './detector/labels/label_qualified.csv'],
          'bbox_path':'../../bbox_result/',
          'preprocessing_backend':'python'
         }
    # stage1相关的path这里都没有用到，可以直接无视
    # luna_raw:(.raw)是3D-CT图像，(.raw)文件和(.mhd)文件置于同一目录
    # luna_segment: 1776个病例的所有分割文件(.mhd)
    # luna_data:(.mhd)是3D-CT图像对应的标签，(.mhd)文件和(.raw)文件置于同一目录
    # preprocess_result_path:预处理后numpy的保存路径，与preprocess_path是同一目录，只是相对于不同的起始父级
    # preprocess_path:预处理后numpy的保存路径，与preprocess_result_path是同一目录，只是相对于不同的起始父级
    # luna_abbr:没用到
    # luna_label:1776个病例的所有标签文件
    # bbox_path:交替训练第1阶段的输出，用于第2阶段的输入之一
    