import numpy as np
import random
import math

#dataSet  已知类别的训练集
#labelSet 类别集
#k        选取的前k个值
#data     数据
def classify(dataSet,k,data):
    #使用欧式距离进行计算
    #得到数据的行数
    rows = len(dataSet)
    dataSet = np.array(dataSet)
    data = np.array(data)
    #rows = int(len(dataSet) - 1)
    distance = np.zeros(rows)
    for i in range(rows):
        sumData = np.sqrt(np.sum(pow(dataSet[i,:-1]-data[:-1],2)))
        #print(sumData)
        distance[i] = sumData
        #for j in range(len(data) - 1):
            #sumData += math.pow((dataSet[i][j] - data[j]),2)
        #distance.append(math.sqrt(sumData))
    #print(distance)
    #distance = np.mat(distance)
    #进行排序
    #print(distance)
    sortedDistance = distance.argsort()
    knnDataSet = []
    for i in range(k):
        knnDataSet.append(dataSet[i])
    return knnDataSet
    
   


