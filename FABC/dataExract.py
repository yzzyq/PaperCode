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
        for i in range(len(dataSet)):
            dataSet[i].append(labelSet[i])            
    elif 'testdata.txt' == fileName:
        with open(path,'r') as f:
            i = 0
            for line in f.readlines():
                i += 1
                data = line.strip().split(' ')
                data = [float(i) for i in data]
                if i > trueNum:
                    data.append(-1)
                else:
                    data.append(1)
                dataSet.append(data)
    return dataSet



#对数据进行规范化
def dataNorm(files_charact):
   maxData = np.max(files_charact)
   minData = np.min(files_charact)
   lenData = maxData - minData 
   for i in range(len(files_charact)):
      for j in range(len(files_charact[i])):
         temp = files_charact[i][j] - minData
         files_charact[i][j] = temp / lenData


def compute(label,dirs):
    correct = 0
    trueNum = 0
    if dirs == 'Hbeta_OIII':
        num = 1635
    else:
        num = 1633
    for i in range(len(label)):
        if i < num and label[i] == 1:
            correct += 1
        if label[i] == 1:
            trueNum += 1
    recall = correct / 1635
    precise = correct / trueNum
    return recall,precise
        
    
