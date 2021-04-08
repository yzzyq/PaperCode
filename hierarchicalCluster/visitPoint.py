#visit point extaction
import numpy as np
import math


def extractionPoint(dataSet,dcThreshold,timeThreshold,tdThreshold):
    #当前的聚类结果
    CC = []
    #之前的聚类结果，之前的结果和当前的结果是有关系的
    PC = []
    #访问点
    VP = []
    #print('dcThreshold:',dcThreshold)
    rowsDataSet = len(dataSet)
    for i in range(rowsDataSet):
        CCcentroid = getCentroid(CC,dataSet)
        PCcentroid = getCentroid(PC,dataSet)
        if distance(dataSet[i],CCcentroid) < dcThreshold:
            CC.append(i)
        else:
            if duration(CC,dataSet) > timeThreshold:
                VP.append(CC)
                CC = []
                PC = [] 
            else:
                if interval(CC,PC,dataSet) > timeThreshold and \
                   distance(CCcentroid,PCcentroid) < tdThreshold:
                    CC = CC + PC
                    VP.append(CC)
                    CC = []
                    PC = []
                else:
                    PC = CC
                    CC = []
    return VP


#经纬度转换为度
def rad(lat):
    return lat*math.pi / 180

#计算该数据与当前簇中心的距离,单位是m            
def distance(data1,data2):
    if data1 is None or data2 is None:
        return 0
    #计算俩个
    earthRadius = 6378137
    radLat1 = rad(data1[0])
    radLat2 = rad(data2[0])
    a = radLat1 - radLat2
    b = rad(data1[1]) - rad(data2[1])
    s = 2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+\
        math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
    s = s*earthRadius
    s = round(s*10000) / 10000
    return s

#计算簇的时间
def duration(cluster,dataSet):
    if len(cluster) < 2:
        return 0
    dataSetArray = np.array(dataSet)
    maxTime = 0
    minTime = float('INF')
    for index in cluster:
        if dataSet[index][2] > maxTime:
            maxTime = dataSet[index][2]
        if dataSet[index][2] < minTime:
            minTime = dataSet[index][2]
    #print('时间：',(maxTime - minTime)*86400)
    return (maxTime - minTime)*86400
    
#计算俩个簇的时间间隔
def interval(cureentC,previousC,dataSet):
    #得出之前簇的最后的时间，当前簇的第一个时间
    pcTime = 0
    for index in previousC:
        if dataSet[index][2] > pcTime:
            pcTime = dataSet[index][2]
    ccTime = float('INF')
    for index in cureentC:
        if dataSet[index][2] < ccTime:
            ccTime = dataSet[index][2]
    return abs(ccTime - pcTime)*86400


#计算出簇的中心,簇中所有
def getCentroid(cluster,dataSet):
    if 0 == len(cluster):
        return None
    allPoint = []
    for index in cluster:
        allPoint.append(dataSet[index][:2])
    return np.mean(allPoint,axis=0)


def getThreshold(dataSet,dcThresholdPro,tdThresholdPro):
    allDis = []
    for i in range(len(dataSet)-1):
        dis = distance(dataSet[i],dataSet[i+1])
        allDis.append(dis)
    allDis = sorted(allDis)
    print(int((len(allDis)+1)*dcThresholdPro))
    dcThreshold = allDis[int((len(allDis))*dcThresholdPro)]
    tdThreshold = allDis[int((len(allDis))*tdThresholdPro)]
    return dcThreshold,tdThreshold

