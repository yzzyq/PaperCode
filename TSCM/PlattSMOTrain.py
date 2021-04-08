import numpy as np
import generalTool
import copy


#SMO的循环运行的过程
def SMOTrain(dataSet,labelSet,C,toler,maxIter):
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labelSet)
    #存储我们所需要的信息
    svm = generalTool.SVM(dataMat,labelMat,C,toler)
    currentIter = 0
    #选取第一个alphas，需要俩种方法混合运行
    entireTrue = True
    alphasChanged = 0
    #前面这个条件就是循环次数超过了，后面这个就是遍历整个集合都没对alphas改变
    while (currentIter < maxIter) and ((alphasChanged > 0) or (entireTrue)):
        alphasChanged = 0
        if entireTrue:
            for i in range(svm.num):
                alphasChanged += innerSmoTrain(i,svm)
            currentIter += 1
        else:
            #非边界alphas
            nonBoundDots = np.nonzero((svm.alphas < C)*(svm.alphas > 0))[0]
            for i in nonBoundDots:
                alphasChanged += innerSmoTrain(i,svm)
            currentIter += 1
        if entireTrue:
            entireTrue = False
        elif alphasChanged == 0:
            entireTrue = True
    return svm.b,svm.alphas

#SMO算法的内部循环
def innerSmoTrain(i,svm):
    errorFirst = generalTool.calcError(svm,i)
    if ((svm.labelSet[0,i]*errorFirst < -svm.toler) and (svm.alphas[i] < svm.C)) \
       or ((svm.labelSet[0,i]*errorFirst > svm.toler) and (svm.alphas[i] < 0)):
        second,secondError = generalTool.choiceSecondAlphas(i,svm,errorFirst)
        alphasFirstCopy = copy.deepcopy(svm.alphas[i])
        alphasSecondCopy = copy.deepcopy(svm.alphas[second])
        if svm.labelSet[0,i] != svm.labelSet[0,second]:
            L = max(0,svm.alphas[second] - svm.alphas[i])
            H = min(svm.C,svm.C + svm.alphas[second] - svm.alphas[i])
        else:
            L = max(0,svm.alphas[second] + svm.alphas[i] - svm.C)
            H = min(svm.C,svm.alphas[second] + svm.alphas[i])
        if L == H:
            return 0
        eta = float(2*svm.dataSetMat[i,:]*svm.dataSetMat[second,:].T - \
              svm.dataSetMat[i,:]*svm.dataSetMat[i,:].T - \
              svm.dataSetMat[second,:]*svm.dataSetMat[second,:].T)
        if eta > 0:
            return 0
        svm.alphas[second] -= svm.labelSet[0,second]*(errorFirst - secondError)/eta
        svm.alphas[second] = generalTool.restrictRange(svm.alphas[second],H,L)
        generalTool.updateError(svm,second)
        if abs(svm.alphas[second] - alphasSecondCopy) < 0.0001:
            return 0
        svm.alphas[i] += svm.labelSet[0,second]*svm.labelSet[0,i]*\
                        (alphasSecondCopy - svm.alphas[second])
        generalTool.updateError(svm,i)
        b1 = svm.b - errorFirst - svm.labelSet[0,i]*\
             (svm.alphas[i] - alphasFirstCopy)\
             *svm.dataSetMat[i,:]*svm.dataSetMat[i,:].T-svm.labelSet[0,second]*\
             (svm.alphas[second] - alphasSecondCopy)*svm.dataSetMat[i,:]*\
             svm.dataSetMat[second,:].T
        
        b2 = svm.b - secondError - svm.labelSet[0,i]*\
             (svm.alphas[i] - alphasFirstCopy)\
             *svm.dataSetMat[i,:]*svm.dataSetMat[second,:].T - \
             svm.labelSet[0,second]*\
             (svm.alphas[second] - alphasSecondCopy)*svm.dataSetMat[second,:]*\
             svm.dataSetMat[second,:].T
        if svm.alphas[i] > 0 and svm.C > svm.alphas[i]:
            svm.b = b1
        elif svm.alphas[second] > 0 and svm.alphas[second] < svm.C:
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2
        return 1
    else:
        return 0

#计算出weights
def calcWeights(alphas,labelSet,dataSet):
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labelSet).T
    m,n = dataMat.shape
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i,0]*labelMat[i,0],dataMat[i,:].T)
    return w
    




            
