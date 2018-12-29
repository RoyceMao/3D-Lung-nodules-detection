# -*- coding: utf-8 -*-
config = {'datapath':'./work/preprocess',
          'preprocess_result_path':'./work/preprocess',
          'outputfile':'prediction.csv',
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':1,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':True, # 跳过预处理过程，这里因为之前已经把所有数据都预处理完毕，所以设为True
         'skip_detect':False} # 跳过第1阶段检测预测过程
