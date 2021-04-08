#根据选中的y来建立core以及它的中心来计算
import kNN
import numpy as np
import math
import copy

'''
def getDistance(x,y):
    sumX = 0
    for i in range(len(x)):
        sumX += pow(x[i] - y[i],2)
    sumX = math.sqrt(sumX)
    return sumX
'''


#得出中心和core
def getCore(knnDataSet,num,y):
    length = len(knnDataSet)
    knnDataSet = np.array(knnDataSet)
    #求得中心
    center = 0
    maxDistance = []
    for i in range(length):
        #论文中的公式无法理解
        #得出最大的K个数
        #distance = getDistance(knnDataSet[i],y)
        distance = np.sqrt(np.sum(pow(knnDataSet[i,:-1]-y[:-1],2)))
        maxDistance.append(distance)
    maxDistance = np.argsort(maxDistance)[::-1]
    minDistance = maxDistance[:num]
    #core的中心
    center = minDistance[num - 1]
    #中心的距离
    minDis = maxDistance[center]
    #求得core
    core = []
    coreData = []
    distances = []
    for i in range(length):
        #distance = getDistance(knnDataSet[i],knnDataSet[center])
        distance = np.sqrt(np.sum(pow(knnDataSet[i,:-1]-knnDataSet[center,:-1],2)))
        distances.append(distance)
    core = np.argsort(distances)
    core = core[0:num]
    for data in core:
        coreData.append(knnDataSet[data])
    return core,center,coreData
    

def getExceptionAndVar(core,num):
    #求出每列的期望
    exception = np.sum(core,axis=0)/(num)
    #求出每列数据的方差
    var = []
    for p in range(len(core[0])):
        variance = 0
        for n in range(len(core)):
            variance += pow(core[n][p] - exception[p],2)
        var.append(variance/len(core))
    exceptionMat = np.mat(exception)
    varMat = np.mat(var)
    return exceptionMat,varMat


def local(dataSet,i,a,k):
    #提出y的值
    y = dataSet[i]
    data = copy.deepcopy(dataSet)
    #使用knn找出点
    del data[i]
    knnDataSet = kNN.classify(data,k,y)
    #找出core和中心了
    core,center,coreData = getCore(knnDataSet,int(a*k),y)
    exceptionMat,varMat = getExceptionAndVar(coreData,a*k)
    coreMat = np.mat(coreData)
    temp = coreMat - exceptionMat
    coreMat = temp / varMat
    U,D,V = np.linalg.svd(coreMat)
    return V,D,core

#求出数据的OD和CD    
def getODandCD(data,D,V,exceptionMat,varMat,num):
    n,p = data.shape
    data = (data - exceptionMat)/varMat
    #求出核心代表值和正交代表值
    coreX = np.dot(V,data.T)
    orthX = data - V.T*coreX
    #求出核心距离和正交距离
    OD = 0
    for i in range(len(orthX)):
        OD += np.dot(orthX[i],orthX[i].T)
    OD = math.sqrt(OD)
    D = 1./D
    D = D.tolist()
    for i in range(p - len(D)):
        D.append(0) 
    DTemp = np.diag(D)
    CD = math.sqrt(np.dot(np.dot(coreX.T,DTemp),coreX)/min(num-1,p))
    return OD,CD

    
#求出数据的权重
def getWeight(CDs,j):
    CDTemp = []
    minCD = float('INF')
    for CD in CDs:
        if CD != 0:
            CDTemp.append(1/CD)
            if minCD > 1/CD:
                minCD = 1/CD
        else:
            CDTemp.append(CD)
    denominator = 0
    for i in range(len(CDTemp)):
        denominator += CDTemp[i] - minCD
    weight = (CDTemp[j] - minCD)/denominator
    return weight

def process(dataSet,i,a):
    k = 10
    #求出i相关的信息
    V,D,argCore = local(dataSet,i,a,k)
    dataSetnoI = np.delete(dataSet,i,axis=0)
    exceptionMat,varMat = getExceptionAndVar(dataSetnoI,len(dataSetnoI))
    #计算每个数据的核心代表值
    dataMat = np.mat(dataSet)
    n,p = dataMat.shape
    CDs = []
    ODs = []
    for j in range(n):
        CD = 0
        OD = 0
        if j != i and j not in argCore:
            #求出OD和CD
            CD,OD = getODandCD(dataMat[j,:],D,V,exceptionMat,varMat,a*k)
        ODs.append(OD)
        CDs.append(CD)
    #求出权重
    locOut = []
    for j in range(n):
        weight = 0
        if not j == i and j not in argCore:
            weight = getWeight(CDs,j)
        locOut.append(weight*ODs[j])
    return locOut
        
        
            
            
    
    
