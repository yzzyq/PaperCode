import dataExract
import os
import time
import numpy as np
import Optics

#时间聚焦聚类
#在各时间段里进行聚类，找出效果最好的那个

#随机选取初始时间间隔I
if not os.path.exists('Result'):
    os.mkdir('Result')
catalog = 'Data'
listdirs = os.listdir(catalog)
for dirs in listdirs:
    path = catalog+'/'+dirs
    listFile = os.listdir(path)
    for file in listFile:
        dataSet = dataExract.extractOneDataSet(path+'/'+file)
        #tGranularity = dataSet[-1][2]/20
        fileName = file.split('.')[0]
        threshold = 0.05
        dataSet = np.array(dataSet)
        minpts = 5
        Eps = 0.1
        startTime = time.clock()
        allDistance = Optics.getAllDistance(dataSet)
        resultList = Optics.opticsProcess(allDistance,minpts,Eps)
        maxClusterResult = Optics.generateCluster(resultList,threshold,minpts)
        clusterResults = []
        for cluster in maxClusterResult:
            oneCluster = []
            for data in cluster:
                oneCluster.append(data[0])
            clusterResults.append(oneCluster)
        endTime = time.clock()
        print(file)
        needTime = '  时间：'+ str(endTime - startTime) +'s'
        #将得到的簇输入到文件中
        fileName = fileName + '.txt'
        if not os.path.exists('Result/' + dirs):
            os.mkdir('Result/' + dirs)
        f = open('Result/'+dirs+'/'+fileName,'w')
        cluster = str(clusterResults) + needTime
        f.write(cluster)
        f.close()
