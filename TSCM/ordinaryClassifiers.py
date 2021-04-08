import dataExract
import kNN
import naiveBayes
import logistReg
import Tree
import PlattSMOTrain as pst
import copy
import random
import numpy as np
#数据在传统分类器中的分类结果


def getOutCome(algoithmNum,trainDataSet,flod):
    #先将数据切分开来
    #1/flod是测试集所占的比例
    trainData = copy.deepcopy(trainDataSet)
    test,train = dataExract.splitDataSet(trainData,1/flod)
    #五种传统的分类器
    results = []
    #KNN
    if algoithmNum == 0:
        for data in test:
            dataType = kNN.classify(train,7,data)
            results.append(dataType)
        return accuracy(test,results)
    #贝叶斯
    elif algoithmNum == 1:
        #得出所有的属性的均值和方差
        dataNormal = naiveBayes.attributesNormal(train)
        for data in test:
            result = naiveBayes.Normal(dataNormal,data)
            results.append(result)
        return accuracy(test,results)
    #逻辑回归
    elif algoithmNum == 2:
        test = np.array(test)
        weights = logistReg.logRegress(train)
        for data in test:
            result = logistReg.sigmoid(np.sum(np.multiply(data[:-1],weights)))
            if result >= 0.5:
                result = 1
            else:
                result = -1
            results.append(float(result))
        return accuracy(test,results)
    #决策树
    elif algoithmNum == 3:
        tree = Tree.buildTree(train)
        for data in test:
            result = Tree.classify(data,tree)
            results.append(result)
        return accuracy(test,results)  
    #SMO
    elif algoithmNum == 4:
        trainLabel = []
        trainSet = copy.deepcopy(train)
        for i in range(len(trainSet)):
            trainLabel.append(trainSet[i][-1])
            del trainSet[i][-1]
        b,alphas = pst.SMOTrain(trainSet,trainLabel,0.6,0.001,50)
        w = pst.calcWeights(alphas,trainLabel,trainSet)
        exeData = np.mat(test)
        m,n = exeData.shape
        for i in range(m):
            result = float(exeData[i,:-1]*np.mat(w) + b)
            if result > 0:
                result = 1
            else:
                result = -1
            results.append(result)
        return accuracy(test,results)

#得出我们的C
def getResults(trainDataSet,choiceDim):
    #C,行就是数据，列就是每个算法的结果
    #每个训练数据在第一步结果在传统机器学习算法中的结果
    col = len(choiceDim)
    n = len(trainDataSet)
    C = np.zeros((n,col))
    #做出五个基础算法的数据集
    train1 = getData(choiceDim[0],trainDataSet)
    train2 = getData(choiceDim[1],trainDataSet)
    train3 = getData(choiceDim[2],trainDataSet)
    train4 = getData(choiceDim[3],trainDataSet)
    train5 = getData(choiceDim[4],trainDataSet)
    for i in range(n):
        #每一个数据进行五个传统算法的检验
        for j in range(col):
            if j == 0:
                train = splitData(train1,i)
                result = kNN.classify(train,7,train1[i])
                C[i,j] = result
            if j == 1:
                train = splitData(train2,i)
                dataNormal = naiveBayes.attributesNormal(train)
                result = naiveBayes.Normal(dataNormal,train2[i])
                C[i,j] = result
            if j == 2:
                train = splitData(train3,i)
                data = np.array(train3[i])
                weights = logistReg.logRegress(train)
                result = logistReg.sigmoid(np.sum(np.multiply(train3[i][:-1],weights)))
                if result < 0.5:
                    result = -1
                else:
                    result = 1
                C[i,j] = result
            if j == 3:
                train = splitData(train4,i)
                tree = Tree.buildTree(train)
                result = Tree.classify(train4[i],tree)
                C[i,j] = result
            if j == 4:
                train = splitData(train5,i)
                trainLabel = []
                trainSet = copy.deepcopy(train)
                for k in range(len(trainSet)):
                    trainLabel.append(trainSet[k][-1])
                trainSet = np.delete(trainSet,-1,axis=1)
                b,alphas = pst.SMOTrain(trainSet,trainLabel,0.6,0.001,50)
                w = pst.calcWeights(alphas,trainLabel,trainSet)
                result = float(np.mat(train5)[i,:-1]*np.mat(w)) + b
                if result > 0:
                    result = 1
                else:
                    result = -1
                
                C[i,j] = result           
    return C
                
                
                
def splitData(train1,i):
    temp = copy.deepcopy(train1)
    length = len(train1)*0.8
    train = []
    while len(train) < length:
        index = int(random.uniform(0,len(train1)))
        if index != i:
            train.append(temp[index])
    return train
                
        
        
    
def getData(choiceDim,trainDataSet):
    trainData = copy.deepcopy(trainDataSet)
    for dim in range(len(choiceDim)-1,-1,-1):
        if choiceDim[dim] == 0:
            for i in range(len(trainDataSet)):
                del trainData[i][dim]
    return trainData
        
        


#查看数据的精确性
def accuracy(exeSet,results):
    correct = 0
    for i in range(len(exeSet)):
        if exeSet[i][-1] == results[i]:
            correct += 1
    return correct/len(results)
        


def testOutCome(data,m,choiceDim,trainDataSet):
    #做出五个基础算法的数据集
    train1 = getData(choiceDim[0],trainDataSet)
    train2 = getData(choiceDim[1],trainDataSet)
    train3 = getData(choiceDim[2],trainDataSet)
    train4 = getData(choiceDim[3],trainDataSet)
    train5 = getData(choiceDim[4],trainDataSet)
    data1 = getTestData(choiceDim[0],data)
    data2 = getTestData(choiceDim[1],data)
    data3 = getTestData(choiceDim[2],data)
    data4 = getTestData(choiceDim[3],data)
    data5 = getTestData(choiceDim[4],data)
    #每一个数据进行五个传统算法的检验
    C = []
    for j in range(m):
        if j == 0:
            result = kNN.classify(train1,7,data1)
        if j == 1:
            dataNormal = naiveBayes.attributesNormal(train2)
            result = naiveBayes.Normal(dataNormal,data2)
        if j == 2:
            data3 = np.array(data3[:-1])
            weights = logistReg.logRegress(train3)
            result = logistReg.sigmoid(np.sum(np.multiply(data3,weights)))
            if result < 0.5:
                result = -1
            else:
                result = 1
        if j == 3:  
            tree = Tree.buildTree(train4)
            result = Tree.classify(data4,tree)
        if j == 4:
            trainLabel = []
            trainSet = copy.deepcopy(train5)
            for i in range(len(trainSet)):
                trainLabel.append(trainSet[i][-1])
            trainSet = np.delete(trainSet,-1,axis=1)
            b,alphas = pst.SMOTrain(trainSet,trainLabel,0.6,0.001,50)
            w = pst.calcWeights(alphas,trainLabel,trainSet)
            result = float(np.mat(data5[:-1])*np.mat(w)) + b
            if result > 0:
                result = 1
            else:
                result = -1
        C.append(result)
    return np.array(C)
    
def getTestData(choiceDim,data):
    dataTemp = copy.deepcopy(data)
    for i in range(len(choiceDim)-1,-1,-1):
        if choiceDim[i] == 0:
            del dataTemp[i]
    return dataTemp
        







        
        
