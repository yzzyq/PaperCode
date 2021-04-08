import copy
import random
import dataExract
import os
import math
import numpy as np

#计算距离
def distance(index1,index2,allDis):
    dis = 0
    time = 0
    dis = abs(allDis[index1][0] - allDis[index2][0])
    time = abs(allDis[index1][1] - allDis[index2][1])
    if time == 0:
        return 0
    return dis/time

#记录全部数据之间的距离
def getAllDistance(dataSet):
    allDis = []
    sumCurrent = 0
    for i in range(len(dataSet)):
        if i == 0:
            allDis.append([0,0,0])
        else:
            #存储的是距离和时间
            dis = []
            tmp = edistance(dataSet[i-1],dataSet[i])
            sumCurrent += tmp
            dis.append(sumCurrent)
            dis.append(dataSet[i][2])
            dis.append(i)
            allDis.append(dis)
    return allDis

#经纬度转换为度
def rad(lat):
    return lat*math.pi / 180

#轨迹距离
def edistance(data1,data2):
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

        
#计算每个核心对象的核心距离
def computeCoreDistance(index,objectPoint,minPts,allDis):
    pointDistances = []
    for i in objectPoint[index]:
        #计算距离
        pointDistance = distance(index,i,allDis)
        pointDistances.append(pointDistance)
    pointDistances.sort()
    return pointDistances[minPts]
    
#计算每个元素的可达距离
def computeReachableDistance(point,coreObject,allDis,coreDistance):
    pointDistance = distance(point,coreObject,allDis)
    return max(pointDistance,coreDistance)

#样本点是否存在队列中
def isExistList(point,objectList,listIndex):
    isExist = False
    for i in range(len(objectList)):
        if objectList[i][0] == point:
            listIndex.append(i)
            isExist = True
            break
    return isExist
            

#对顺序队列进行排序    
def sortOrder(orderList):
    for i in range(len(orderList)-1):
        for j in range(len(orderList)-i-1):
            if orderList[j][1] > orderList[j+1][1]:
                orderList[j],orderList[j+1] = orderList[j+1],orderList[j]
    return orderList

#删除核心对象中相应的元素列表
def delete(coreObject,firstElement):
    for i in range(len(coreObject)):
        if coreObject[i][0] == firstElement:
            del coreObject[i]
            break
    return coreObject
            
def searchIndex(firstElement,coreObject):
    index = 0
    for i in range(len(coreObject)):
        if coreObject[i][0] == firstElement:
            index = i
            break
    return index

def opticsProcess(allDis,minPts,radius):
    #核心对象和它的核心距离
    coreObject = []
    #顺序队列
    orderList = []
    #结果队列
    resultList = []
    #每个对象的范围内的点
    objectPoint = []
    #找出所有的核心对象
    for i in range(len(allDis)):
        points = []
        for j in range(len(allDis)):
            if i != j:
                pointDistance = distance(i,j,allDis)
                if pointDistance < radius:
                    points.append(j)
        objectPoint.append(points)
        if len(points) > minPts:
            coreObject.append(i)
    while len(coreObject) > 0:
        #随机选择一个核心对象
        #coreRandom = random.randrange(0,len(coreObject))
        coreRandom = 0
        index = coreObject[coreRandom]
        order = []
        #计算出这个核心对象的核心距离
        coreDistance = computeCoreDistance(index,objectPoint,minPts,allDis)
        #放在顺序队列中,每一个元素放的是点和可达距离
        order.append(index)
        order.append(coreDistance)
        order.append(allDis[index][2])
        orderList.append(order)
        #对顺序队列进行循环处理
        while len(orderList) > 0:
            firstElement = orderList[0]
            #放入结果队列中去
            resultList.append(firstElement)
            del orderList[0]
            firstElementIndex = firstElement[0]
            #拓展这个元素
            if len(objectPoint[firstElementIndex]) > minPts \
                  and (firstElementIndex in coreObject):
                coreDistance1 = computeCoreDistance(firstElementIndex,objectPoint,minPts,allDis)
                #删除核心对象中相应的元素列表
                coreObject.remove(firstElementIndex)
                for i in objectPoint[firstElementIndex]:
                    #样本点既不在结果队列也不在顺序队列中
                    order = []
                    listIndex = []
                    if not isExistList(i,resultList,listIndex) \
                       and not isExistList(i,orderList,listIndex):
                        #计算每个元素的可达距离
                        reachableDistance = computeReachableDistance(\
                            i,firstElementIndex,allDis,coreDistance1)
                        order.append(i)
                        order.append(reachableDistance)
                        order.append(allDis[i][2])
                        orderList.append(order)
                        #对顺序队列进行排序
                        orderList = sortOrder(orderList)
                    #样本点在顺序队列中不在结果队列中 
                    elif not isExistList(i,resultList,listIndex) \
                         and isExistList(i,orderList,listIndex):
                        index = listIndex[0]
                        #比较新旧距离
                        reachableDistance = computeReachableDistance(\
                            i,firstElementIndex,allDis,coreDistance1)
                        #如果旧的小于新的，那么代替旧的    
                        if orderList[index][1] > reachableDistance:
                            order.append(i)
                            order.append(reachableDistance)
                            order.append(allDis[i][2])
                            orderList[index] =  order
                            #对顺序队列进行排序
                            orderList = sortOrder(orderList)
            #不可拓展的点直接放入orderList中
            elif len(objectPoint[firstElementIndex]) <= minPts and \
                 not isExistList(i,resultList,listIndex) and \
                 not isExistList(i,orderList,listIndex):
                order = []
                reachableDistance = computeReachableDistance(\
                            i,firstElementIndex,allDis,coreDistance)
                order.append(i)
                order.append(reachableDistance)
                order.append(allDis[i][2])
                orderList.append(order)
                orderList = sortOrder(orderList)
    return resultList

#生成簇
def generateCluster(resultList,threshold,minPits):
    cluster = []
    separations = []
    resultIndex = []
    for result in resultList:
        tmp = []
        tmp.append(result[-1])
        tmp.append(result[1])
        resultIndex.append(tmp)
    #得到离群点
    for k in range(len(resultIndex)):
        if resultIndex[k][1] >= threshold:
            separations.append(k)
    #如果没有离群点，全部都在一个簇中
    if len(separations) == 0:
        cluster.append(resultIndex)
    else:
        start = separations[0]
        cluster.append(resultIndex[0:start])
        end = 0
        for i in range(len(separations)-1):
            start = separations[i]
            end = separations[i+1]
            cluster.append(resultIndex[start+1:end])
        cluster.append(resultIndex[end+1:len(resultIndex)])
    for j in range(len(cluster)-1,-1,-1):
        if len(cluster[j]) < minPits:
            del cluster[j]
    return cluster                   
        
        
        
    
    
    
 
