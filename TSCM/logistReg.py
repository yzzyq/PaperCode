import numpy as np
import copy
import random
#逻辑回归

def sigmoid(data):
    if data >= 0:
        return 1/(1 + np.exp(-data))
    else:
        return np.exp(data)/(1+np.exp(data))

    
#numIter是迭代次数
def logRegress(dataSet,numIter=5):
    dataMat = np.mat(dataSet)
    m,n = dataMat.shape
    #初始化权重值
    weights = np.ones((1,n-1))
    for i in range(numIter):
        dataCopy = copy.deepcopy(dataMat)
        for j in range(m):
            step = 4 / (1 + i + j) + 0.01
            index = random.randint(0,dataCopy.shape[0])
            #算出它的值
            #print(np.sum(np.multiply(dataCopy[j,:-1],weights)))
            result = sigmoid(np.sum(np.multiply(dataCopy[j,:-1],weights)))
            error = dataSet[j][-1] - result
            weights = weights + step*error*dataCopy[j,:-1]
            np.delete(dataCopy,j,axis = 0)
    return weights

    
