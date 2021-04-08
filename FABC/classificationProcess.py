#分类的不同步骤算法的实现过程
import numpy as np

#计算隶属度
def objectMembership(data,num,clusterCenter,temp,m):
    #拿出需要用到的簇中心
    cluster = clusterCenter[num]
    value = abs(data - cluster)**(2/(m-1))
    value = value / temp
    if value == 0:
        value = 1
    return 1/(value)

def allCluster(data,clusterCenter,clusterNum,m):
    sumData = 0
    for i in range(clusterNum):
        temp = abs(data - clusterCenter[i])
        sumData += temp**(2/(m-1))
    return sumData
    
#相似度的计算
#col是数据的列数，weight就是权重
def SimilarityEvaluation(exeData,trainData,col,weights,clusterCenter,m): 
    #初始化相似度计算
    sim = 0
    orderWeights = []
    k = len(clusterCenter[0])
    for i in range(col):
        sindF = 0
        #和每个簇中心进行计算
        exeTemp = allCluster(exeData[i],clusterCenter[i],k,m)
        trainTemp = allCluster(trainData[i],clusterCenter[i],k,m)
        for j in range(k):
            sindF += min(objectMembership(exeData[i],j,clusterCenter[i],exeTemp,m),\
                         objectMembership(trainData[i],j,clusterCenter[i],trainTemp,m))
        sim += weights[i]*sindF     
        orderWeights.append(sindF)
    orderWeights.sort()
    sim = sum(list(map(lambda a,b:a*b,orderWeights,weights)))
    return sim

#类别之间的隶属度
def classMembership(classNum,data,weights,clusterCenter,m,trainData,col,num):
    allSim = 0
    classSim = 0
    for i in range(num):
        temp = SimilarityEvaluation(data,trainData[i],col,weights,\
                                       clusterCenter,m)
        allSim += temp
        if trainData[i][-1] == classNum:
            classSim += temp
    temp = (classSim / allSim)*0.49
    if data[-1] == classNum:
        return 0.51 + temp
    else:
        return temp
        
#与每个训练数据进行比较
#classNum类别数量
def VotingValues(trainData,exeData,weights,classNum,clusterCenter):
    #训练数据的个数
    m = 2
    num = len(trainData)
    col  = len(trainData[0]) - 1
    v = np.zeros((num,classNum))
    for i in range(num):
        #与每个训练数据计算相似度
        sim = SimilarityEvaluation(exeData,trainData[i],col,weights,\
                                   clusterCenter,m)
        for j in range(classNum):
            if j == 0:
                classJ = -1
                temp = classMembership(classJ,trainData[i],weights,\
                                   clusterCenter,m,trainData,col,num)
            else:
                classJ = j
                temp = classMembership(classJ,trainData[i],weights,\
                                   clusterCenter,m,trainData,col,num)
            v[i,j] = sim*temp
    return v

#决策最后的一个公式
def computeNum(classI,i,T,trainData):
    sumV = 0
    n = len(trainData)
    sumNum = 0
    for j in range(n):
        if trainData[j][-1] == classI:
            sumNum += 1
            sumV += T[j,i]
    temp = (1/(sumNum))*sumV
    #反函数
    return temp

#分类过程
def decisionProcess(trainData,exeData,weights,classNum,clusterCenter):
    T = VotingValues(trainData,exeData,weights,classNum,clusterCenter)
    results = []
    for i in range(classNum):
        if i == 0:
            classI = -1
        else:
            classI = i
        result = computeNum(classI,i,T,trainData)
        results.append(result)
    result = np.argmax(results)
    if result == 0:
        result = -1
    return result
    
        


            
        
