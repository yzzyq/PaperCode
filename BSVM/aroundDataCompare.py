import numpy as np

#周边数据的比较
def getAroundData(dataProcess,allPoistion,threshold,label):
    lenPoistion = len(allPoistion)
    #记录相互靠近的光谱
    aroundData = []
    for i in range(lenPoistion):
        NegPoistion = []
        #找出所有邻近的点
        classNum = 0
        for j in range(lenPoistion):
            if i != j:
                dis = np.sqrt(np.sum(pow(allPoistion[i] - allPoistion[j],2)))
                if dis < threshold:
                    NegPoistion.append(j)
        aroundData.append(NegPoistion)
    updateAroundData(dataProcess,aroundData,label)  
    return aroundData

#根据周围的类别更新数据的类别
def updateAroundData(dataProcess,aroundData,label):
    lenData = len(dataProcess)
    for i in range(lenData):
        num = 0
        for j in aroundData[i]:
            # if dataProcess[j,-1] == 1:
            # num += dataProcess[j,-1]
            num += label[j]
        dataProcess[i,2] = num
