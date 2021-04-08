#主要的运行代码
import visitPoint
import visitPointCluster
import dataExract
import os
import time



if not os.path.exists('Result'):
    os.mkdir('Result')
catalog = 'Data'
listdirs = os.listdir(catalog)
#设置初始阈值
dcThresholdPro = 0.97
timeThreshold = 300
tdThresholdPro = 0.994
eps = 50
for dirs in listdirs:
    path = catalog +'/'+dirs+'/'+'Trajectory'
    listFile = os.listdir(path)
    for file in listFile:
        dataSet = dataExract.extractOneDataSet(path+'/'+file)
        dcThreshold,tdThreshold = visitPoint.getThreshold(dataSet,\
                                    dcThresholdPro,tdThresholdPro)
        
        fileName = file.split('.')[0]
        startTime = time.clock()
        VP = visitPoint.extractionPoint(dataSet,dcThreshold,timeThreshold,\
                                        tdThreshold)
        CS = visitPointCluster.vpCluster(VP,dataSet,eps)
        endTime = time.clock()
        needTime = '  时间：' + str(endTime - startTime) + 's'
        print(file)
        #将得到的簇输入到文件中
        fileName = fileName + '.txt'
        if not os.path.exists('Result/' + dirs):
            os.mkdir('Result/' + dirs)
        f = open('Result/'+dirs+'/'+fileName,'w')
        cluster = str(CS)
        cluster += needTime
        f.write(cluster)
        f.close()
