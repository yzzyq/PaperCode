#训练过程
#1.先用k-means生成p个landmark
#2.建立landmark和点之间的landmark矩阵
#3.进行奇异值分解
#4.V的每一行就是一个数据，利用k-means再聚类
import dataExract as dataSE
import KAver
import numpy as np
import math

#高斯核
def gaussianKernel(x,z,reachRate):
    temp = x - z
    return np.exp(np.dot(temp,temp.T) / (-2*(reachRate**2)))

#映射矩阵
def generateProjectMatrix(dataSetMat,landmark,markAssment,reachRate):
    m,n = landmark.shape
    num,col = dataSetMat.shape
    W = np.mat(np.zeros((col,num)))
    for i in range(num):
        clusterNum = int(markAssment[i][0])
        sumProjection = np.sum(gaussianKernel(dataSetMat[i,:],\
                                              landmark[clusterNum],reachRate))
        for j in range(n):
            temp = gaussianKernel(dataSetMat[i,j],landmark[clusterNum,j],\
                                  reachRate)
            W[j,i] = temp / sumProjection
    m,n = W.shape
    for i in range(m):
        sumLine = np.sum(W[i,:])
        W[i,:] = W[i,:] / math.pow(sumLine,1/2)
    return W
            
#奇异值分解出的V
def generateV(baseValue,baseVector,W):
    num = baseVector.shape[0]
    baseValue = [1/value for value in baseValue]
    if len(baseValue) < num:
        addMat = np.mat(np.zeros((num-len(baseValue),len(baseValue))))
    eigenValuesMat = np.diag(baseValue)
    eigenValuesMat = np.row_stack((eigenValuesMat,addMat))
    temp = np.dot(eigenValuesMat,baseVector.T)
    return np.dot(temp,W)

def train(dataSet,reachRate,firstK,clusterNum):
    dataSetMat = np.mat(dataSet)
    #簇中的个数
    #k = 2
    landmark,markAssment = KAver.KMeans(dataSet,clusterNum,\
                                    KAver.getDistance,KAver.createCenter)
    W = generateProjectMatrix(dataSetMat,landmark,markAssment,reachRate)
    #计算它的特征值和特征向量
    eigenValues,FeatureVector = np.linalg.eig(np.dot(W,W.T))
    #将特征值降序排列
    sortValue = np.argsort(-eigenValues)
    sortValue = sortValue[:firstK]
    eigenValues = -np.sort(-eigenValues)
    #选出前K个特征值和对应的特征向量
    baseValue = eigenValues[:firstK]
    baseVector = FeatureVector[:,sortValue]
    #print(baseValue,baseVector)
    #计算V
    V = generateV(baseValue,baseVector,W)
    dataSE.dataNorm(V.T)
    clusterCenter,clusterAssment = KAver.KMeans(V.T.tolist(),clusterNum,\
                                    KAver.getDistance,KAver.createCenter)
    return clusterCenter,clusterAssment

