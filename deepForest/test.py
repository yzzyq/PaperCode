import numpy as np
from gcforest import *
from multi import *
from largeThreeLayer import *
from time import time
import csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import numpy as np
# import pandas as pd
# pd.set_option('display.max_columns',1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth',1000)

def readCsv(file_name):
    labelSet = []
    dataSet = []    
    reader = csv.reader(open(file_name,'r',encoding='utf-8'))
    for row_data in reader:
        # print(row_data)
        # row_data = row_data.strip('')
        row_data = [one_row_data for one_row_data in row_data if one_row_data != '']
        # labelSet.append(float(row_data[-1]))
        labelSet.append(row_data[-1])
        # print('row_data:', row_data)
        if row_data[0] == '\ufeff12':
            row_data[0] = 12     
        data = [float(row_data[index]) for index in range(len(row_data) - 1)]

        dataSet.append(data)
    dataSet_mat = np.array(dataSet)
    # label_set = set(labelSet)
    # print(label_set)
    # print(dataSet)
    # print(labelSet)
    return dataSet, labelSet


if __name__ == '__main__':
    dataSet, labelSet = readCsv('data1.csv')
   
    print('dataSet:', len(dataSet))
    print('labelSet:', len(labelSet))
    classification = len(set(labelSet))
    print('类别数：',classification)
    # aa = aa
    # 多粒度度扫描,这里窗口的大小最好是d/16, d/8, d/4
    windows_num = [len(dataSet[0]) // 4,len(dataSet[0]) // 6,len(dataSet[0]) // 2]
    # windows_num = [len(dataSet[0]) // 4]
    print('window_num:', windows_num)
    mult_grain = Multi_Grained(100, 2, windows_num)
    
    # 窗口扫描，窗口大小是5
    win, win_label, win_len = mult_grain.window(dataSet, labelSet)
    # print(win)
    # print(win_label)
    print(win_len)
    print('移动窗口结束')
    # win, win_label, win_len = mult_grain.window(5,dataSet, labelSet)
    # val_proba, win_label = mult_grain.train(win, win_label, win_len)
    mult_grain.train(win, win_label)
    all_transform_data = mult_grain.getTransformData(win, win_len)
    all_transform_label = []
    for one_win_len_index,one_win_len in enumerate(win_len):
        all_transform_label.append(labelSet)
    # print('多粒度扫描完成，得到转换后的数据：', len(all_transform_data),len(all_transform_data[0]),len(all_transform_data[0][0]))
    # print(all_transform_data[0])
    # print('------------------------')
    # print(all_transform_data[1])
    # print('-------------------------')
    # print(all_transform_data[2])
    # 进入级联森林,训练森林
    clf = gcforest(num_estimator= 100, num_forests = 4, classification_num = classification, 
                   one_level_layer_num = len(windows_num), max_depth= 100, max_level= 2, n_fold= 5)
    clf.train(all_transform_data, all_transform_label)
    print('级联森林训练结束')
    #测试数据
    dataSet, labelSet = readCsv('test1.csv')
    print('---------------数据测试开始-----------------')
    win, win_label, win_len = mult_grain.window(dataSet, labelSet)
    all_transform_data = mult_grain.getTransformData(win, win_len)
    result = clf.predict(all_transform_data)
    label_set = set(labelSet)
    sort_label = sorted(label_set) 
    print('最终结果:', result)   
    result = [sort_label[one_result] for one_result in result]
    print('最终结果:', result)
    print("accuracy_score:", accuracy_score(labelSet, result))    
    print("precision_score", metrics.precision_score(labelSet, result, average='macro'))
    print("f1_score", metrics.f1_score(labelSet, result, average='macro'))

