import random
import numpy as np
import Optics
import math
import copy
#时间聚焦聚类
#在各时间段里进行聚类，找出效果最好的那个

def tfoptics(dataSet,tGranularity,threshold):
    allDistance = Optics.getAllDistance(dataSet)
    dataSet = np.array(dataSet)
    minpts = 5
    Eps = 0.1
    maxClusterResult = []
    firstR = 0
    resultList = Optics.opticsProcess(dataSet,minpts,Eps)
    cluster = Optics.generateCluster(resultList,threshold,minpts)
    return 
    #i = 0
    '''
    while firstR == 0:
        #time1 = random.randint(0,7)
        #time2 = random.randint(14,20)
        startTime = 0
        endTime = 20
        timeInterval = [startTime*tGranularity,endTime*tGranularity]
        #needDataSet = getNeedDataSet(timeInterval,allDistance)
        #resultList = Optics.opticsProcess(needDataSet,minpts,Eps)
        resultList = Optics.opticsProcess(dataSet,minpts,Eps)
        cluster = Optics.generateCluster(resultList,threshold,minpts)
        firstR = getR(cluster)
        if firstR == 0:
            Eps += Eps
            threshold += threshold
        if threshold > 1:
            return [],[]
    '''
        #i += 1
    #print('firstR:',firstR)
    #初始进行比较
    #isIter,currentTimeInterval,maxClusterResult = getMaxValue(firstR,\
            #tGranularity,minpts,Eps,timeInterval,threshold,allDistance)
    #while isIter:
        #timeInterval = currentTimeInterval
        #isIter,currentTimeInterval,maxClusterResult = getMaxValue(firstR,\
            #tGranularity,minpts,Eps,timeInterval,threshold,allDistance)
    #return currentTimeInterval,maxClusterResult

#获取几个中最大价值的那个时间段
def getMaxValue(firstR,tGranularity,minpts,Eps,timeInterval,threshold,allDistance):
    allTimeInterval = []
    allTimeInterval.append(timeInterval)
    allTimeInterval.append([min(timeInterval[0]+tGranularity,timeInterval[1]),\
                            max(timeInterval[0]+tGranularity,timeInterval[1])])
    allTimeInterval.append([min(max(allDistance[0][1],timeInterval[0]-tGranularity),timeInterval[1]),\
                            max(max(allDistance[0][1],timeInterval[0]-tGranularity),timeInterval[1])])
    allTimeInterval.append([min(timeInterval[0],min(allDistance[-1][1],timeInterval[1]+tGranularity)),\
                            max(timeInterval[0],min(allDistance[-1][1],timeInterval[1]+tGranularity))])
    allTimeInterval.append([min(timeInterval[0],timeInterval[1]-tGranularity),\
                            max(timeInterval[0],timeInterval[1]-tGranularity)]) 
    maxValue = 0
    maxTime = []
    isIter = False
    maxClusterResult = []
    for i in range(len(allTimeInterval)):
        needDataSet = getNeedDataSet(allTimeInterval[i],allDistance)
        if 0 == i:
            maxTime = allTimeInterval[i]
            maxValue,maxClusterResult = getValue(firstR,allTimeInterval[i],\
                            threshold,needDataSet,minpts,Eps)
        else:
            currentTime = allTimeInterval[i]
            currentValue,clusterResult = getValue(firstR,allTimeInterval[i],\
                            threshold,needDataSet,minpts,Eps)
            if currentValue > maxValue and currentValue!= 0:
                isIter = True
                maxValue = currentValue
                maxTime = allTimeInterval[i]
                maxClusterResult = clusterResult
    return isIter,maxTime,maxClusterResult

#得到R
def getR(cluster):
    if len(cluster) == 0:
        return 0
    allSumCluster = 0
    num = 0
    for data in cluster:
        for i in data:
            num += 1
            allSumCluster += i[1]
    R = allSumCluster / num
    return R

#获取这个时间段的价值
def getValue(firstR,timeInterval,threshold,dataSet,minpts,Eps):
    resultList = Optics.opticsProcess(dataSet,minpts,Eps)
    cluster = Optics.generateCluster(resultList,threshold,minpts)
    R = getR(cluster)
    if R != 0:
        thresholdNews = (R / firstR)*threshold
        #使用更新后的R进行聚类
        cluster = Optics.generateCluster(resultList,thresholdNews,minpts)
        Q1 = -getR(cluster)
        Q2 = Q1 / math.log(10 + abs(timeInterval[1] - timeInterval[0]))
        return Q2,cluster
    else:
        return 0,[]
    
#获取这个时间的数据集
def getNeedDataSet(timeIntervalIndex,allDistance):
    allDis = []
    length = len(allDistance)
    for i in range(length):
        if allDistance[i][1] >= timeIntervalIndex[0] and allDistance[i][1] <= timeIntervalIndex[1]:
            tmp = copy.deepcopy(allDistance[i])
            tmp.append(i)
            allDis.append(tmp)
    return allDis
    
