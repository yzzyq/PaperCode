import numpy as np
import random
import math

#dataSet  已知类别的训练集
#labelSet 类别集
#k        选取的前k个值
#data     数据
def classify(dataSet,k,data,labelSet):
    #使用欧式距离进行计算
    #得到数据的行数
    rows = int(len(dataSet))
    distance = []
    for i in range(rows):
        sumData = 0
        for j in range(len(data)):
            sumData += math.pow((dataSet[i][j] - data[j]),2)
        distance.append(math.sqrt(sumData))
    distance = np.mat(distance)
    #进行排序
    sortedDistance = distance.argsort()
    #取出距离最小的k个值
    dataFea = 0
    unDataFes = 0
    #确定前K个点所在类别的出现频率
    for i in range(k):
        index = int(sortedDistance[0,i])
        if labelSet[index] == 1:
            dataFea += 1
        elif labelSet[index] == -1:
            unDataFes += 1
    #返回前k个点出现频率最高的类别作为当前点的预测分类
    if dataFea > unDataFes:
        return 1
    else:
        return -1

