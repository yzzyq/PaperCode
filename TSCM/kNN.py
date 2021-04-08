import numpy as np
import random
import math
#k-近邻算法:1.计算已知类别的点和当前点的之间距离
#           2.按照距离递增次序排序  
#           3.选取与当前点最小的k个点
#           4.确定前K个点所在类别的出现频率
#           5.返回前k个点出现频率最高的类别作为当前点的预测分类


#dataSet  已知类别的训练集
#k        选取的前k个值
#data     数据
def classify(dataSet,k,data):
    #使用欧式距离进行计算
    #得到数据的行数
    rows = int(len(dataSet) - 1)
    distance = []
    for i in range(rows):
        sumData = 0
        for j in range(len(data) - 1):
            sumData += math.pow((dataSet[i][j] - data[j]),2)
        distance.append(math.sqrt(sumData))
    distance = np.mat(distance)
    #进行排序
    sortedDistance = distance.argsort()
    #取出距离最小的k个值
    dataFea = 0
    unDataFes = 0
    #确定前K个点所在类别的出现频率
    #print('sortedDistance:',sortedDistance)
    for i in range(k):
        #print(i)
        index = int(sortedDistance[0,i])
        if dataSet[index][-1] == 1:
            dataFea += 1
        elif dataSet[index][-1] == -1:
            unDataFes += 1
    #返回前k个点出现频率最高的类别作为当前点的预测分类
    if dataFea > unDataFes:
        return 1
    else:
        return -1
   

