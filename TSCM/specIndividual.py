import random
import numpy as np
import ordinaryClassifiers as oc
import copy
#种群中的个体

class speciesIndividual:

    def __init__(self,num,genesNum):
        #基因序列
        self.genes = np.ones(genesNum)
        self.num = num
        self.genesNum = genesNum
        #个体的适应度
        self.fitness = 0
        if 2 == self.num:
            self.genes = [random.randint(0,10) for i in range(self.genesNum)]
            sumGen = sum(self.genes)
            self.genes = [gene/sumGen for gene in self.genes]

    #计算第一阶段的适应度
    def calndividualFitness(self,num,algoithmNum,trainDataSet,flod):
        trainData = copy.deepcopy(trainDataSet)
        for i in range(self.genesNum-1,-1,-1):
            if self.genes[i] == 0:
                for j in range(len(trainDataSet)): 
                    del trainData[j][i]
        results = []
        for i in range(flod):
            result = oc.getOutCome(algoithmNum,trainData,flod)
            results.append(result)
        self.fitness = sum(results)/flod

    #计算第二阶段的适应度
    def calSecondFitness(self,C,labelSet):
        results = []
        length = len(C)
        for i in range(length):
            CE = np.sum(C[i]*self.genes)
            if CE <= 0:
                CE = -1
            elif CE > 0:
                CE = 1
            results.append(CE)
        correct = 0
        for i in range(length):
            if labelSet[i] == results[i]:
                correct += 1
        self.fitness = correct / length
            
