import numpy as np
import random
import math
#k-近邻算法:1.计算已知类别的点和当前点的之间距离
#           2.按照距离递增次序排序  
#           3.选取与当前点最小的k个点
#           4.确定前K个点所在类别的出现频率
#           5.返回前k个点出现频率最高的类别作为当前点的预测分类


#dataSet  已知类别的训练集
#labelSet 类别集
#k        选取的前k个值
#data     数据

class KnnAlogrithm:
    def __init__(self, k, dataSet, labelSet):
        self.k = k
        self.dataSet = dataSet
        self.labelSet = labelSet
        self.is_write = 1
    
    def fit(self, data, label):
        self.dataSet = data
        self.labelSet = label
    
    def predict(self, dataSet):
        all_result = []
        all_conn = []
        for data in dataSet:
            one_result,one_conn = self.predictData(data)
            all_result.append(one_result)
            all_conn.append(one_conn)
        return all_result,all_conn

    def predictData(self, data):
        
        #使用欧式距离进行计算
        #得到数据的行数
        rows = len(self.dataSet)
        distance = []
        for i in range(rows):
            sumData = 0
            for j in range(len(data)):
                sumData += math.pow((self.dataSet[i][j] - data[j]),2)
            distance.append(math.sqrt(sumData))
        #print(distance)
        distance = np.mat(distance)
        #进行排序
        #print(distance)
        sortedDistance = distance.argsort()
        #print(sortedDistance)
        #取出距离最小的k个值
        dataFea = 0
        unDataFes = 0
        #确定前K个点所在类别的出现频率
        k_min = min(len(sortedDistance),self.k)
        for i in range(k_min):  
            index = int(sortedDistance[0,i])
            if self.labelSet[index] == 1:
                dataFea += 1
            elif self.labelSet[index] == -1:
                unDataFes += 1
        #返回类别和置信度
        if dataFea > unDataFes:
            return 1,dataFea / (dataFea + unDataFes)
        else:
            return -1,unDataFes / (dataFea + unDataFes)
   
