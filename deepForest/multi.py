#输入原始特性向量
#我的原始数据是用线指数算出来的400个线指数，数组形式[1,2,...400]
#用滑动窗口将特性向量输出
import os
import copy
import numpy as np
import math
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import pandas as pd
import csv


#定义多粒度扫描类
class Multi_Grained:

    # 初始化
    def __init__(self, n_estimators, num_forests, scan_size, max_depth = 100, min_samples_leaf = 1):
        #定义森林数
        self.num_forests = num_forests
        #定义每个森林的树个数
        self.n_estimators = n_estimators
        #每棵树的最大深度
        self.max_depth = max_depth
        #树会生长到所有叶子都分到一个类，或者某节点所代表的样本数已小于min_sample_leaf
        self.min_samples_leaf = min_samples_leaf
        # 全部扫描窗口的大小
        self.scan_size = np.array(scan_size) 
        # 扫描窗口的个数
        self.scan_num = len(scan_size)
        #最后产生的类向量
        self.model = []
    


  

    # 定义滑动窗口函数
    def window(self, dataSet, labelSet):
        win, win_label = [], []
        win_len = len(dataSet[0])- self.scan_size + 1
        for windowsize_index,windowsize in enumerate(self.scan_size):
            one_window_all_data = []
            one_window_all_label = []
            for index,data in enumerate(dataSet):
                # print(data)
                # print(index)
                # 得到所有分开的数据,保存下来        
                for i in range(win_len[windowsize_index]):
                    one_window_all_data.append(data[i:i+windowsize])
                    one_window_all_label.append(labelSet[index])
            win.append(one_window_all_data)
            # print(one_window_all_data)
            win_label.append(one_window_all_label)
            # print(one_window_all_label)
        return win,win_label,win_len

    # 训练模型    
    def train(self, win, win_label):
        for one_model_index in range(self.scan_num):
            one_model = []
            clf = RandomForestClassifier(n_estimators= self.n_estimators,
                                        n_jobs= -1,
                                        max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf)                
            clf.fit(win[one_model_index],win_label[one_model_index])
            one_model.append(clf)

            clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                    n_jobs= -1,
                                    max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf)
            clf.fit(win[one_model_index],win_label[one_model_index])   
            one_model.append(clf)
            self.model.append(one_model)
        
    # 得到我们的转换数据
    def getTransformData(self, win, win_len):
        all_transform_data = []
        # all_transform_label = []
        for one_win_len_index,one_win_len in enumerate(win_len):
            data_num = int(len(win[one_win_len_index]) / one_win_len)
            # print('数据的个数:',data_num)
            transform_data = []
            # all_transform_label.extend(labelSet)
            for current_num in range(data_num):
                start = current_num*one_win_len
                end = (current_num + 1)*one_win_len
                # print(start,end)
                one_result = []
                for one_model in self.model[one_win_len_index]:
                    one_one_result = one_model.predict_proba(win[one_win_len_index][start:end])
                    # print('one_one_result',one_one_result)
                    # one_result = np.concatenate((one_result,one_one_result),axis = 0)
                    one_one_result = one_one_result.tolist()
                    for one in one_one_result:
                        one_result.extend(one)
                        
                # aa = aa
                transform_data.append(one_result)
            all_transform_data.append(transform_data)
        # print('len(transform_data)',len(transform_data))
        return all_transform_data



    
                


         
