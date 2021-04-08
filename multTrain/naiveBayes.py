#贝叶斯算法
import csv
import random
import math
import copy
from decimal import Decimal

class GaussianNB:
    def __init__(self):
        self.dataNormal = {}
        self.is_write = 1

    def fit(self, train, label):
        self.dataNormal = self.attributesNormal(train, label)
        

    #按照类别划分数据，这里就是0和1
    def splitDataSetByClass(self, dataSet, label):
        dataClass = {}
        for index,data in enumerate(dataSet):
            key = label[index]
            if key not in dataClass.keys():
                dataClass[key] = [] 
            dataClass[key].append(data)
        return dataClass

    #因为数据中的数值都是连续型的
    #计算均值
    def averageData(self, dataSet):    
        return sum(dataSet)/len(dataSet)
        
    #计算方差
    def varianceData(self, dataSet):
        aver = self.averageData(dataSet)
        temp = 0.0
        dataSetTemp = dataSet
        for data in dataSet:
            temp += math.pow(data - aver,2)+1
        return math.sqrt(temp/(len(dataSet)+2))

    #计算每个属性的均值和方差
    def attributesNormal(self, dataSet, labelSet):
        #按照类别划分数据
        dataClass = self.splitDataSetByClass(dataSet, labelSet)
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
                aver = self.averageData(dataCol)
                variance = self.varianceData(dataCol)
                attri.append(aver)
                attri.append(variance)
  
                dataNormal[dataAttri].append(attri)

        return dataNormal

    #计算每个属性高斯密度函数
    def normalFunction(self, value, data):
        aver = value[0]
        variance = value[1]
        num = 0
        temp = math.exp(-(float)(math.pow(data-aver,2))/(2*math.pow(variance,2)))
        return (1/math.sqrt(2*(math.pi)*variance))*temp


    #计算每个类别的每个属性密度函数值
    def predict(self, dataSet):
        #比较俩个类别，谁大就是谁
        all_label = []
        all_score = []
        for exeData in dataSet:
            bestLabel = None
            bestScoer = -float('inf')
            for key in self.dataNormal.keys():   
                values = self.dataNormal[key]
                normalClass = 1
                for i in range(len(values)):
                    #注意0的情况
                    PrioriProbability = float(self.normalFunction(values[i],exeData[i]))
                    normalClass *= PrioriProbability
                    # if PrioriProbability != 0 and normalClass != 0:
                    #     #浮点数计算
                    #     normalClass *= Decimal(PrioriProbability)
                if normalClass > bestScoer:
                    bestScoer = normalClass
                    bestLabel = key    
            all_score.append(bestScoer)
            all_label.append(bestLabel) 
        return all_label,all_score     
        # return bestLabel,bestScoer
