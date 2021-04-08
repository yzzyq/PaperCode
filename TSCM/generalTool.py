import random
import numpy as np
import csv
#提取数据集
def loadDataSet(filename):
    dataSet = []
    labelSet = []
    with open(filename) as file:
        File = csv.reader(file,delimiter=',')
        data = list(File)
        for i in range(len(data)):
            dataSet.append([float(data[i][0]),float(data[i][1])/50])
            if float(data[i][-1]) == 0:
                data[i][-1] = -1
            labelSet.append(int(data[i][-1]))
    return dataSet,labelSet

#将数据集划分为我们所需的训练集和测试集
def splitDataSet(dataSet,labelSet,ratio):
    num = int(len(dataSet)*ratio)
    exeDataSet = []
    exeLabel = []
    while len(dataSet) > num:
        randI = random.randrange(len(dataSet))
        exeDataSet.append(dataSet[randI])
        exeLabel.append(labelSet[randI])
        del dataSet[randI]
        del labelSet[randI]
    return exeDataSet,exeLabel



#存储必要的信息
class SVM:
    def __init__(self,dataSetMat,labelSet,C,toler):
        #数据和类别 
        self.dataSetMat = dataSetMat
        self.labelSet = labelSet
        #软间隔
        self.C = C
        #容错率
        self.toler = toler
        #数据的数量
        self.num = dataSetMat.shape[0]
        #要求的俩个系数，也就是边界
        self.alphas = np.zeros((self.num,1))
        self.b = 0
        #每个数据的错误率
        self.eCache = np.zeros((self.num,2))

#计算误差
def calcError(svm,k):
    #计算算出的值
    result = float(np.multiply(svm.alphas,svm.labelSet.T).T*\
                   (svm.dataSetMat*svm.dataSetMat[k,:].T)) + svm.b
    error = result - float(svm.labelSet[0,k])
    return float(error)

#选出第二个alphas来更新,选出步长最大的那个
def choiceSecondAlphas(k,svm,errorFirst):
    maxK = 0
    maxStep = 0
    maxError = 0
    svm.eCache[k] = [1,errorFirst]
    #选出有效的误差缓存
    validErrorEcache = np.nonzero(svm.eCache[:,0])[0]
    if(len(validErrorEcache) > 1):
        for errorIndex in validErrorEcache:
            #自己就不用相减比较了
            if errorIndex == k:
                continue
            errorTemp = calcError(svm,errorIndex)
            step = abs(errorTemp - errorFirst)
            if step > maxStep:
                maxK = errorIndex
                maxStep = step
                maxError = errorTemp
        return maxK,maxError
    else:
        index = k
        while index == k:
            index = int(random.uniform(0,svm.num))
        error = calcError(svm,index)
        return index,error

#更新误差缓存
def updateError(svm,index):
    error = calcError(svm,index)
    svm.eCache[index] = [1,error]
    

#将之限制在范围内
def restrictRange(num,top,bottom):
    if num > top:
        num = top
    elif num < bottom:
        num = bottom
    return num

