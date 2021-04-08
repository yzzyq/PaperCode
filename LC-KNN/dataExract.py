import numpy as np
import os
import random

#提取数据
def extractData(location_name,dirs,fileName):
    dataSet = []
    labelSet = []
    trueNum = 0
    path = location_name + fileName
    if dirs == 'Hbeta_OIII':
        trueNum = 1635
    else:
        trueNum = 1633 
    if 'traindata.txt' == fileName:
        with open(path,'r') as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                data = [float(i) for i in data]
                dataSet.append(data)
    
        with open(location_name + 'trainlabel.txt','r') as f:
            for line in f.readlines():
                label = int(float(line.strip()))
                labelSet.append(label)
                    
    elif 'testdata.txt' == fileName:
        if dirs == 'Hbeta_OIII':
            trueNum = 1635
        else:
            trueNum = 1633 
        with open(path,'r') as f:
            i = 0
            for line in f.readlines():
                i += 1
                data = line.strip().split(' ')
                data = [float(i) for i in data]
                dataSet.append(data)
                if i > trueNum:
                    labelSet.append(-1)
                else:
                    labelSet.append(1)
    
    dataNorm(dataSet)
    return dataSet,labelSet

#对数据进行规范化
def dataNorm(files_charact):
   maxData = np.max(files_charact)
   minData = np.min(files_charact)
   lenData = maxData - minData 
   for i in range(len(files_charact)):
      for j in range(len(files_charact[i])-1):
         temp = files_charact[i][j] - minData
         files_charact[i][j] = temp / lenData
    

#计算召回率和精确度
def computeRecall(results,dirs,labelSet):
    #得到结果中正类别的个数
    correct = 0
    trueNum = 0
    lenResults = len(results)
    for i in range(lenResults):
        if 1 == labelSet[i] and 1 == results[i]:
            correct += 1
        if 1 == results[i]:
            trueNum += 1
    print('correct:',correct)
    print('trueNum:',trueNum)
    if dirs == 'Hbeta_OIII':
        #数据中正的个数是1635
        #召回率
        recall = correct/1635
        precise = correct/trueNum
    else:
        #其他俩个的正的个数是1633
        recall = correct/1633
        precise = correct/trueNum
    return recall,precise
    
    
