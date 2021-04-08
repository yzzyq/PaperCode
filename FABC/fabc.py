#FABC
import numpy as np
import time
import fcm
import classificationProcess as cp
import dataExract
import os



if not os.path.exists('Result'):
    os.mkdir('Result')
if not os.path.exists('ResultRun'):
    os.mkdir('ResultRun')
testCatalog = 'test'
trainCatalog = 'train'
listdirs = os.listdir(trainCatalog)
for dirs in listdirs:
    print(dirs)
    trainPath = trainCatalog+'/'+dirs+'/'
    testPath = testCatalog+'/'+dirs+'/'
    #获取训练数据和测试数据
    trainDataSet = dataExract.extractData(trainPath,dirs,'traindata.txt')
    exeDataSet = dataExract.extractData(testPath,dirs,'testdata.txt')
    #print('训练数据的个数：',len(trainDataSet))
    #print('测试数据的个数：',len(exeDataSet))
    #print('训练数据的维度：',len(trainDataSet[0]))
    #print('测试数据的维度：',len(exeDataSet[0]))
    classlist = []
    for data in trainDataSet:
        classlist.append(data[-1])
    classNum = len(set(classlist))
    trainMat = np.mat(trainDataSet)
    #对训练数据的每维数据进行FCM处理
    print('对数据进行FCM处理。。')
    clusterCenters,weights = fcm.dimProcess(trainMat[:,:-1])

    #拿出测试数据
    results = []
    print('拿出测试数据。。')
    exelabelSet = []
    i = 0
    for data in exeDataSet:
        print(i)
        exelabelSet.append(data[-1])
        #进行决策
        result = cp.decisionProcess(trainDataSet,data[:-1],weights,classNum,clusterCenters)
        results.append(result)
        i += 1
    
    print('得出测试结果')
    trueResults = []
    falseResults = []
    for i in range(len(results)):
        if results[i] == 1:
            trueResults.append(i)
        else:
            falseResults.append(i)
            
    cluster = str('正类别：') + str(trueResults) \
                  + str('    负类别：') + str(falseResults)

    fileName = dirs + '.txt'
    f = open('ResultRun/'+fileName,'w')
    f.write(cluster)
    f.close()
    
    recall,precise = dataExract.compute(results,dirs)
    print('得出召回率：',recall)
    print('得出精确度：',precise)
    
    #将得到的簇输入到文件中

    f = open('Result/'+fileName,'w')
    cluster = '召回率：' + str(recall) + '    精确度：' + str(precise)
    f.write(cluster)
    f.close()

