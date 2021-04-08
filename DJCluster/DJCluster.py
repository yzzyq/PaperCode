import os
import dataExract
import numpy as py
import math
from astropy.io import fits
import numpy as np
import time
#DJCluster
#density-and-join-based algorithm

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


# 一一处理文件夹中的数据
def process(catalog,minpts,Eps):
    listdirs = os.listdir(catalog)
    for dirs in listdirs:
        listFile = os.listdir(catalog+'//'+dirs+'//'+'Trajectory')
        for file in listFile:
            dataSet = dataExract.extractOneDataSet(file)
            fileName = dirs.split('.')[0]
            #使用DJCluster处理数据
            cluster = getClusterByDJCluster(dataSet,minpts,Eps)
            #将得到的簇输入到文件中
            with open(dirs+'//'+fileName+'.txt','w') as f:
                f.write(cluster)

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
            if list_num >= 9000:
                break
        index += 1
    
    return np.array(all_data),all_label,all_file


#经纬度转换为度
def rad(lat):
    return lat*math.pi / 180

#根据经纬度计算俩个点的距离的
def distance(data1,data2):
    earthRadius = 6378137
    radLat1 = rad(data1[0])
    radLat2 = rad(data2[0])
    a = radLat1 - radLat2
    b = rad(data1[1]) - rad(data2[1])
    s = 2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+\
        math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
    s = s*earthRadius
    s = round(s*10000) / 10000
    return s

#开始使用聚类处理数据
def getClusterByDJCluster(dataSet,minpts,Eps):
    dataLen = len(dataSet)
    #初始化所有未访问过的数据
    unvisited = np.zeros(dataLen)
    cluster = []
    noise = []
    #全部遍历
    for i in range(dataLen):
        #如果没有簇，那么可以处理
        if unvisited[i] == 0:
            unvisited[i] = 1
            densityNeighbohood = getDensityNeighbohood(dataSet,i,minpts,Eps)
            mix = [i for i in range(len(cluster)) if len(list(set(\
                densityNeighbohood).intersection(set(cluster[i])))) > 0]
            #对于簇的三种处理
            if len(densityNeighbohood) == 0:
                noise.append(i)
            elif len(mix) > 0:
                cluster[mix[0]] = list(set(cluster[mix[0]].extend(\
                    densityNeighbohood)))
            else:
                cluster.append(densityNeighbohood)
            for node in densityNeighbohood:
                unvisited[node] = 1
    return cluster
        
#找出这个点的邻域    
def getDensityNeighbohood(dataSet,index,minpts,Eps):
    neighborhood = []
    for i in range(len(dataSet)):
        if i != index:
            dis = distance(dataSet[i],dataSet[index])
            if dis <= Eps:
                neighborhood.append(i)
    if len(neighborhood) < minpts:
        neighborhood = []
    return neighborhood





if __name__ == '__main__':
    # process('Data',10,5)
    start_time = time.clock()
    dataMat, label, all_file = exractData()
    minpts = 10
    Eps = 5
    cluster = getClusterByDJCluster(dataMat,minpts,Eps)
    recall,precise,acc = getRecallAndPrecise(cluster,label)
    # print(cluster)
    end_time = time.clock()
    print('时间：', end_time - start_time)
    print('得到召回率和精确率')
    print('recall:',recall)
    print('precise:',precise)
    print('acc:',acc)
    
