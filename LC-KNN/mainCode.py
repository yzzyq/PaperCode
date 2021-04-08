import dataExract as dataSE
import testingProcess as testPro
import trainingProcess as trainPro
import time
import os


if not os.path.exists('Result'):
    os.mkdir('Result')
if not os.path.exists('Time'):
    os.mkdir('Time')
testCatalog = 'test'
trainCatalog = 'train'
listdirs = os.listdir(trainCatalog)
for dirs in listdirs:
    print(dirs)
    trainPath = trainCatalog+'/'+dirs+'/'
    #整个过程
    #拿出训练数据
    dataSet,labelSet = dataSE.extractData(trainPath,dirs,'traindata.txt')
    
    print('提取出训练样本',len(dataSet),len(labelSet))
    print('训练样本的维度：',len(dataSet[0]))
    reachRate = 2
    firstK = 10
    clusterNum = 2
    startTime = time.clock()
    clusterCenter,clusterAssment = trainPro.train(dataSet,reachRate,firstK,\
                                                  clusterNum)
    print(clusterCenter)
    print(clusterAssment)
    print('训练结束')
    #拿出测试数据
    testPath = testCatalog+'/'+dirs+'/'
    exeDataSet,exelabelSet = dataSE.extractData(testPath,dirs,'testdata.txt')
    print('提取出训练样本',len(exeDataSet),len(exelabelSet))
    print('训练样本的维度：',len(exeDataSet[0]))
    results = testPro.test(exeDataSet,dataSet,clusterCenter,clusterAssment,\
                           labelSet)
    endTime = time.clock()
    needTime = '时间：'+ str(endTime - startTime) +'s'
    with open('Time/'+dirs+'_time.txt','w') as f:
        f.write(needTime)

    for i in range(len(results)):
        if results[i] == 1:
            with open('Result/'+dirs+'_positive_category.txt','a') as f:
                f.write(str(i)+'\n')
        else:
            with open('Result/'+dirs+'_negative_category.txt','a') as f:
                f.write(str(i)+'\n')      
    
