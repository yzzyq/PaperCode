import random
import math
import copy
import randomTree
from collections import Counter
import numpy as np

class RandomTrees:

    def __init__(self):
        self.tree_num = 2
        self.forest = []
        self.is_write = 1

    #训练
    def fit(self, trainSet, labelSet):
        #建立每棵决策树
        for i in range(self.tree_num):
            #首先选出数据集
            #print(labelSet)
            childData,childLabel,chidFeatures = self.randomSample(trainSet,labelSet)
            print('chidFeatures:',chidFeatures)
            #建立决策树
            childTree = randomTree.buildTree(childData,childLabel,chidFeatures)
            self.forest.append(childTree)

    #又放回的随机选取数据和特征
    def randomSample(self,dataSet,labelSet):
        lenData = len(dataSet)
        if len(dataSet[0]) > 1:
            logLenFeature = len(dataSet[0]) - 1
        labelCopy = [col_index for col_index in range(len(dataSet[0]))]
        # print('labelCopy:',labelCopy)
        # print('len(dataSet[0]):',len(dataSet[0]))
        print('dataSet:',len(dataSet))
        print('labelSet:',len(labelSet))
        #子树的数据集
        childData = []
        childLabel = []
        #选取特征
        childFeatures = random.sample(labelCopy,logLenFeature)
        # while len(childFeatures) < logLenFeature:
        #     #print('labelSet长度：',len(labelSet))
        #     randomNum = random.randint(0,len(labelCopy)-1)
        #     childFeatures.append(labelCopy[randomNum])
        #     np.delete(labelCopy, randomNum)
        #print(childFeatures)
        
        #有放回的选取数据
        while len(childData) < lenData:
            randomNum = random.randint(0,lenData-1)
            #print('randomNUM',randomNum)
            data = []
            for num in childFeatures:
                #print(num)
                #print('dataSet',len(dataSet[randomNum]))
                #print(dataSet[randomNum][num])
                data.append(dataSet[randomNum][num])
            childLabel.append(labelSet[randomNum])
            childData.append(data)
        return np.array(childData),np.array(childLabel),np.array(childFeatures)

    #预测
    def predict(self, dataSet):
        all_result = []
        all_confes = []
        for data in dataSet:
            results = []
            for tree in self.forest:
                result,confidence = randomTree.classify(data,tree)
                results.append([list(result.keys())[0],confidence])
            feaNum = 0
            fes_con = 0
            noFeaNum = 0
            no_fes_con = 0
            resultData = 1
            #采用投票法，票数最多的就是类别
            for i in range(len(results)):
                #print('result[i]',results[i])
                if results[i][0] == 1:
                    feaNum += 1
                    fes_con += results[i][1]
                else:
                    noFeaNum += 1
                    no_fes_con += results[i][1]
            result_con = fes_con
            if feaNum < noFeaNum:
                resultData = -1
                result_con = no_fes_con
            all_result.append(resultData)
            all_confes.append(result_con)
        return all_result,all_confes

