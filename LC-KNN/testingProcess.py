#测试过程
#1.将m个簇心和测试数据距离比较
#2.选出最短的那个簇心
#3.将测试数据与此簇心进行KNN算法
import kNN
import KAver
import numpy as np
import math

def getDistance(dataSet,clusterCenter):
    sumData = 0
    clusterCenter = clusterCenter[0]
    for i in range(len(dataSet)):
        temp = math.pow(dataSet[i] - clusterCenter[i],2)
        sumData += temp
    return math.sqrt(sumData)

#找到最短距离的簇心
def findShortCluster(data,clusterCenter):
    shortDistanceCluster = 0
    minDistance = np.inf
    for i in range(len(clusterCenter)):
        distance = getDistance(data,clusterCenter[i].tolist())
        if distance < minDistance:
            minDistance = distance
            shortDistanceCluster = i
    return shortDistanceCluster

#根据簇心找到数据
def findData(dataSet,shortDistanceCluster,clusterAssment,labelSet):
    cluster = []
    cluster = [i for i in range(len(clusterAssment)) \
               if clusterAssment[i][0] == shortDistanceCluster]
    temData = [dataSet[i] for i in cluster]
    temLabel = [labelSet[i] for i in cluster]
    return temData,temLabel
    

def test(exeDataSet,dataSet,clusterCenter,clusterAssment,labelSet):
    results = []
    #print(clusterCenter)
    for data in exeDataSet:
        shortDistanceCluster = findShortCluster(data,clusterCenter)
        cluster,temLabel = findData(dataSet,shortDistanceCluster,\
                                    clusterAssment,labelSet)
        
        dataType = kNN.classify(cluster,10,data,temLabel)
        print(dataType)
        results.append(dataType)
    return results
        
        
        
    
