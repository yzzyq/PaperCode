#FCM
#1.初始化所需要的值，簇个数、权重、隶属度
#2.根据初始化的值和推导出的公式，计算簇中心
#3.根据得出的各种值更新隶属度
#4.不断地进行迭代，满足结束条件则推出循环
import numpy as np


#初始化隶属度，一个数据的全部隶属度为1
def initializaMembership(n,clusterNum):
    #隶属度
    membership = []
    #初始化
    for i in range(n):
        randTemp = np.random.randint(1,n,clusterNum)
        randTemp = [ j / sum(randTemp) for j in randTemp]
        membership.append(randTemp)
    #print('初始化:',membership)
    return np.mat(membership)
    
#计算簇中心
def calcClusterCenter(membershipMat,n,dataMat,clusterNum,m):
    #簇中心
    cluster_centers = []
    n,col = membershipMat.shape
    for i in range(clusterNum):
        for j in range(n):
            membershipMat[j,i] = membershipMat[j,i]**m
        allColMembership = np.sum(membershipMat[:,i]) 
        sumCenter = 0
        for j in range(n):
            sumCenter += membershipMat[j,i] * dataMat[j,0]
        result = sumCenter / allColMembership
        cluster_centers.append(result)
    return np.mat(cluster_centers)

def allCluster(data,clusterCenter,clusterNum,m):
    sumData = 0
    #print('data:',data.tolist())
    data = data.tolist()
    for i in range(clusterNum):
        #print('i:',i)
        temp = abs(data[0] - clusterCenter[0,i])
        sumData += temp**(2/(m-1))
    #print(sumData)
    return sumData
    
    
#更新隶属度
def updateMembership(clusterCenter,n,clusterNum,dataMat,m):
    membership = []
    for i in range(n):
        sumCluster = allCluster(dataMat[i,:],clusterCenter,clusterNum,m)
        member = []
        for j in range(clusterNum):
            data = abs(dataMat[i,0] - clusterCenter[0,j])**(2/(m-1))
            #print('data:',data)
            data = (data / sumCluster)
            if data == 0:
                data += 1
            member.append((1/data)[0])
        membership.append(member)
    #print('membership:',np.mat(membership))
    return np.mat(membership)

#目标函数
def targets(dataMat,clusterCenter,membershipMat,m):
    n,clusterNum = membershipMat.shape
    value = 0
    for i in range(clusterNum):
        for j in range(n):
            temp = pow(dataMat[j,0] - clusterCenter[0,i],2)
            value += pow(membershipMat[j,i],m)*temp
    return value


    
#运行的主程序
def fuzzyCMeans(dataMat):
    #dataMat = np.mat(dataSet)
    n,dim = dataMat.shape

    #初始化各值
    #隶属度因子，轻缓程度
    m = 2.5
    #分类簇数
    clusterNum = 2
    #最大迭代数
    maxIter = 5

    #阈值
    minThreshold = 10
    
    #隶属度，加起来是1
    membershipMat = initializaMembership(n,clusterNum)
    numIter = 0

    #根据现有的条件计算出簇中心
    clusterCenter = calcClusterCenter(membershipMat,n,dataMat,clusterNum,m)
    
    #print(clusterCenter)
    evenValue = 0
    oddValue = 0
    #进行迭代
    while numIter <= maxIter:
        #更新隶属度
        membershipMat = updateMembership(clusterCenter,n,clusterNum,dataMat,m)
        
        #根据现有的条件计算出簇中心
        clusterCenter = calcClusterCenter(membershipMat,n,dataMat,clusterNum,m)

        #计算目标函数的价值
        value = targets(dataMat,clusterCenter,membershipMat,m)

        if numIter % 2 == 0:
            evenValue = value
        else:
            oddValue = value
        #print('---------------------------------------------------------------')
        #print('evenValue:',evenValue)
        #print('oddValue:',oddValue)

        if numIter > 1:
            if abs(evenValue - oddValue) < minThreshold:
                break
       
        numIter += 1
    #得出结果
    #print('membershipMat:',membershipMat)    
    results = membershipMat.argmax(axis=1)
    #print('results:',results)
    #print('clusterCenter:',clusterCenter)
    #print('clusterNum',clusterNum)
    return results,clusterCenter,clusterNum

#权重的划分
def weightsDivision(trainData):
    n,col = trainData.shape
    weights = []
    for i in range(col):
        w = (i/col) - (i-1)/col
        weights.append(w)
    return weights

def weightsDivision1(trainData):
    n,col = trainData.shape
    weights = []
    for i in range(col):
        w = 1
        weights.append(w)
    return weights

def dimProcess(dataMat):
    n,col = dataMat.shape
    clusterCenters = []
    for i in range(col):
        result,clusterCenter,clusterNum = fuzzyCMeans(dataMat[:,i])
        clusterCenters.append(clusterCenter.tolist()[0])
    weights = weightsDivision(dataMat)
    return clusterCenters,weights
    


    

if __name__ == '__main__':
    c = [[1,2],[1,3],[4,5],[3,4]]
    dataMat = np.mat(c)
    n,col = dataMat.shape
    clusterCenters = []
    for i in range(col):
        print(dataMat[:,i])
        result,clusterCenter,clusterNum = fuzzyCMeans(dataMat[:,i])
        clusterCenters.append(clusterCenter)
        print("===========================================================")
    print(clusterCenters)
        
    
