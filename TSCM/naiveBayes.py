#贝叶斯算法
import random
import math
import copy
from decimal import Decimal
import numpy as np

#按照类别划分数据，这里就是0和1
def splitDataSetByClass(dataSet):
    dataClass = {}
    for data in dataSet:
        key = data[-1]
        if key not in dataClass.keys():
            dataClass[key] = []
        #del data[-1]    
        dataClass[key].append(data)
    dataSet = np.delete(dataSet,-1,axis=1)
    return dataClass

#因为数据中的数值都是连续型的
#计算均值
def averageData(dataSet):    
    return sum(dataSet)/len(dataSet)
    
#计算方差
def varianceData(dataSet):
    aver = averageData(dataSet)
    temp = 0.0
    dataSetTemp = dataSet
    for data in dataSet:
        temp += math.pow(data - aver,2)+1
    return math.sqrt(temp/(len(dataSet)+2))

#计算每个属性的均值和方差
def attributesNormal(dataSet):
    #按照类别划分数据
    dataClass = splitDataSetByClass(dataSet)
    dataNormal = {}
    #每个类别进行循环
    for dataAttri in dataClass.keys():
        data = dataClass[dataAttri]
        dataNormal[dataAttri] = []
            
        #每列元素组合在一起
        dataAttribute = zip(*data)
        
        #计算每列的均值和方差
        for dataCol in dataAttribute:      
            attri = []  
            aver = averageData(dataCol)
            variance = varianceData(dataCol)
            attri.append(aver)
            attri.append(variance)
            dataNormal[dataAttri].append(attri)

    return dataNormal

#计算每个属性高斯密度函数
def normalFunction(value,data):
    aver = value[0]
    variance = value[1]
    num = 0
    temp = math.exp(-(float)(math.pow(data-aver,2))/(2*math.pow(variance,2)))
    return (1/math.sqrt(2*(math.pi)*variance))*temp


#计算每个类别的每个属性密度函数值
def Normal(dataNormal,exeData):
    bestLabel = None
    bestScoer = 0.0
    #比较俩个类别，谁大就是谁
    for key in dataNormal.keys():   
        values = dataNormal[key]
        normalClass = 1
        for i in range(len(values)):
            #注意0的情况
            PrioriProbability = float(normalFunction(values[i],exeData[i]))
            if PrioriProbability != 0 and normalClass != 0:
                #浮点数计算
                normalClass *= Decimal(PrioriProbability)
        if normalClass > bestScoer:
            bestScoer = normalClass
            bestLabel = key      
    return bestLabel

    


