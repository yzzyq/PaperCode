import os
from astropy.io import fits
import numpy as np
import visitPoint
import visitPointCluster
import time


#得出我们的召回率和精确率
def getRecallAndPrecise(cluster,labelSet):
    three_class = {}
    num = 0
    for label in labelSet:
        if label not in three_class.keys():
            three_class[label] = 1
            num += 1
        else:
            three_class[label] += 1
    # print('num:',num)
    recall,precise = 0,0
    all_true_num = 0
    #簇中哪类数量最多，那么就是哪一类
    for one_cluster in cluster:
        one_cluster_three = {}
        for index in one_cluster:
            if labelSet[index] not in one_cluster_three:
                one_cluster_three[labelSet[index]] = 1
            else:
                one_cluster_three[labelSet[index]] += 1
        # print('one_cluster_three:',one_cluster_three)
        cluster_class = max(one_cluster_three,key = one_cluster_three.get)
        all_true_num += one_cluster_three[cluster_class]
        recall += one_cluster_three[cluster_class] / three_class[cluster_class]
        precise += one_cluster_three[cluster_class] / len(one_cluster)
        # print('recall:',recall)
        # print('precise',precise)
        # print('----------------------')
    recall /= num
    precise /= num
    acc = all_true_num / len(labelSet)
    return recall,precise,acc

#对单个光谱数据的处理
def readfits(path,fileName):
    dfu = fits.open(path + '/'+fileName)
    #初始波长
    beginWave = dfu[0].header['COEFF0']
    #步长
    step = dfu[0].header['CD1_1']
    #光谱中的流量 
    flux = np.array(dfu[0].data[0])
    #求出波长,求出与流量对应的波长
    wave = np.array([10**(beginWave + step*j) for j in range(len(flux))])
    data = [wave,flux]
    #-------------------------------------------
    return flux[:3000]

#数据文件中的光谱数据
def exractData():
    # 先读取恒星数据
    # all_file_name = ['D:/zhaoguowei/G5.7/解压/A5V-1']
    # all_file_name = ['D:/zhaoguowei/G5.7/解压/A5V-1', 'D:/zhaoguowei/G5.7/解压/f5-1',
    #                  'D:/zhaoguowei/G5.7/解压/G5-1', 'D:/zhaoguowei/G5.7/解压/K5-1']
    all_file_name = ['D:/zhaoguowei/G5.7/解压/A5V-1', 'D:/zhaoguowei/G5.7/解压/K5-1']
    index = 0
    all_file = []
    all_label = []
    all_data = []
    for file_name in all_file_name:
        listFile = os.listdir(file_name)
        list_num = 0
        for file_index, one_file in enumerate(listFile):
            # print(one_file)
            all_file.append(one_file)
            temp_data = readfits(file_name,one_file)
            # print(temp_data)
            all_data.append(temp_data)
            all_label.append(index)
            list_num += 1
            if list_num >= 10000:
                break
        index += 1
    
    return np.array(all_data),all_label,all_file

if __name__ == '__main__':
    startTime = time.clock()
    #设置初始阈值
    dcThresholdPro = 0.95
    timeThreshold = 200
    tdThresholdPro = 0.94
    eps = 50
    dataSet, label, all_file = exractData()
    dcThreshold,tdThreshold = visitPoint.getThreshold(dataSet,\
                                dcThresholdPro,tdThresholdPro)
    VP = visitPoint.extractionPoint(dataSet,dcThreshold,timeThreshold,\
                                    tdThreshold)
    CS = visitPointCluster.vpCluster(VP,dataSet,eps)
    endTime = time.clock()
    needTime = '  时间：' + str(endTime - startTime) + 's'
    #将得到的簇输入到文件中
    # print(CS)
    print(needTime)
    recall,precise,acc = getRecallAndPrecise(CS,label)
    print('得到召回率和精确率')
    print('recall:',recall)
    print('precise:',precise)
    print('acc:',acc)