import fitsData
import selfDataCompare as sdc
import lineDataCompare as ldc
import aroundDataCompare as adc
import copy
import math
import numpy as np
from sklearn import svm

def normalData(train_data):
    for col_num in range(len(train_data[0])):
        min_col = min(train_data[:,col_num])
        max_col = max(train_data[:,col_num])
        gap = max_col - min_col
        train_data[:,col_num] = (train_data[:,col_num] - min_col)/gap
    return train_data

def getLabel(len_data):
    true_index = [158, 530, 589, 1274, 3018, 3953, 4006, 321, 381, 529, 
                  1769, 1829, 3382, 3885, 3980, 4216, 4393, 4620, 312, 
                  1365, 2086, 2110, 2294, 2694, 2784, 2952, 3846]
    # true_index = [1,4,6,8,45,78,100,200,300,400]
    train_label = [-1]*len_data
    for index in true_index:
        train_label[index - 1] = 1
    return train_label


#将fits文件中的数据变成三列
def getData(file_name, side, center, dis_threshold):
    dataSet,allPoistion = fitsData.exractData(file_name)
    dataProcess = np.zeros((len(dataSet), 3))
    label = sdc.getFluxData(dataSet,side,center,dataProcess)

    #查看线性表中的其他发射线的情况
    ldc.getOtherEmissionLine(dataSet,dataProcess,side)

    #查看周边数据的情况
    adc.getAroundData(dataProcess,allPoistion,dis_threshold,label)
    return dataProcess


#在循环中计算中数据的精确率
def getAccuracy(test_result, true_class):
    num = len(test_result)
    sum_true = sum([test_result[index] == true_class[index] for index in range(num)])
    return sum_true / num

#得到测试数据的精确率和召回率
def getTestAccAndRecall(test_class, test_true_index):
    test_true_index = list(map(lambda x:x-1, test_true_index))
    true_index = sum([test_class[index] == 1 for index in test_true_index])
    neg_true_Index = sum([test_class[index] == -1 \
                          for index in range(len(test_class)) \
                          if index not in test_true_index])
    return (true_index + neg_true_Index) / len(test_class), true_index / len(test_true_index)

# #得到相关的点
# def getPoint(train, label, SVMTrainAccuracy, acc_threshold, c, g):
#     is_getRelevantPoints = False
#     relevant_point_index = set({})
#     #检查数据是否被删除
#     train_index = set([num for num in range(len(train))])
#     while not is_getRelevantPoints:
#         train_index_list = list(train_index)
#         #生成未保存的数据
#         train_dummy = [train[index] for index in train_index_list]
#         #使用prim对数据建立最小生成树
#         neighbour = getMSTbyPrim(train_dummy)
#         #先找出基础点，相连并且属于不同类别的点
#         for one_index,neg_point_index in enumerate(neighbour):
#             # is_exist = [label[train_index_list[one_index]] == label[train_index_list[neg_index]] for neg_index in neg_point_index]
#             # #如果其中的俩个点是不同类别的话，那么它们就是基础点
#             # if False in is_exist:
#             #     [relevant_point_index.add(train_index_list[i]) for i,x in enumerate(is_exist) if not x]
#             #     relevant_point_index.add(train_index_list[one_index])
#             # for neg_index in neg_point_index:
#             if label[train_index_list[int(one_index)]] != label[train_index_list[int(neg_point_index)]]:
#                 # print('relevant_point_index:',relevant_point_index)
#                 # print('{int(one_index),int(neg_point_index)}:',{int(one_index),int(neg_point_index)})
#                 relevant_point_index = relevant_point_index | {int(one_index),int(neg_point_index)}
        
#         #根据相关系数得到相关数据
#         relevant_point = [train[index] for index in relevant_point_index]
#         relevant_label = [label[index] for index in relevant_point_index]

#         ms_svm_model = svm.SVC(C = c, kernel= 'rbf' , gamma = g)
#         ms_svm_model.fit(relevant_point, relevant_label)
#         train_result = ms_svm_model.predict(train)

#         MSSVMTrainAccuracy = getAccuracy(train_result, label)
#         if SVMTrainAccuracy - MSSVMTrainAccuracy > acc_threshold:
#             #将已经放入的数据去除
#             train_index -= relevant_point_index
#         else:
#             is_getRelevantPoints = True
#     relevant_point = [train[index] for index in relevant_point_index]
#     relevant_label = [label[index] for index in relevant_point_index]
#     return relevant_point,relevant_label


#得到相关的点
def getPoint(train, label, SVMTrainAccuracy, acc_threshold, c, g):
    is_getRelevantPoints = False
    relevant_point_index = set({})
    #检查数据是否被删除
    train_index = set([num for num in range(len(train))])
    # while not is_getRelevantPoints:
    train_index_list = list(train_index)
    #生成未保存的数据
    train_dummy = [train[index] for index in train_index_list]
    #使用prim对数据建立最小生成树
    neighbour = getMSTbyPrim(train_dummy)
    #先找出基础点，相连并且属于不同类别的点
    for one_index,neg_point_index in enumerate(neighbour):
        # #如果其中的俩个点是不同类别的话，那么它们就是基础点
        if label[train_index_list[int(one_index)]] != label[train_index_list[int(neg_point_index)]]:
            relevant_point_index = relevant_point_index | {int(one_index),int(neg_point_index)}
    before_relevant = set({})
    while not is_getRelevantPoints:    
        #根据相关系数得到相关数据
        # print('relevant_point_index:',relevant_point_index)
        relevant_point = [train[index] for index in relevant_point_index]
        relevant_label = [label[index] for index in relevant_point_index]

        ms_svm_model = svm.SVC(C = c, kernel= 'rbf' , gamma = g)
        ms_svm_model.fit(relevant_point, relevant_label)
        train_result = ms_svm_model.predict(train)

        MSSVMTrainAccuracy = getAccuracy(train_result, label)
        if SVMTrainAccuracy - MSSVMTrainAccuracy > acc_threshold:
            #将已经放入的数据去除
            train_index -= relevant_point_index
            old_points = copy.deepcopy(relevant_point_index)
            relevant_point_index = updateRelevantPoint(before_relevant,relevant_point_index,neighbour)
            before_relevant = old_points
        else:
            is_getRelevantPoints = True

    relevant_point = [train[index] for index in relevant_point_index]
    relevant_label = [label[index] for index in relevant_point_index]
    return relevant_point,relevant_label

def updateRelevantPoint(before_relevant,relevant_point_index,neighbour):
    add_point = relevant_point_index - before_relevant
    for point in add_point:
        relevant_point_index.add(int(neighbour[point]))
    return relevant_point_index


#prim生成最小生成树算法
def getMSTbyPrim(train):
    len_train = len(train)
    train = np.array(train)
    dis_array = np.zeros((len_train,len_train))
    #建造邻近矩阵sadad
    for one_index in range(len_train):
        for two_index in range(len_train):
            # print(train[one_index,:] - train[two_index,:])
            dis_array[one_index,two_index] = math.sqrt(np.sum(pow(train[one_index,:] - train[two_index,:],2)))
    #查看是否访问过
    is_visit = np.zeros(len_train)
    #每个点相邻的点
    neighbour = np.zeros(len_train)
    #存储每个点到树的最短距离,开始的时候，直接选取第一个点
    dis_tree = dis_array[0,:]
    # min_dis_point_tree = getPointToPointDis(train[0,:],train[1:,:])
    is_visit[0] = 1
    while 0 in is_visit:
        minDis = float('inf')
        minDis_pos = -1
        for index in range(len_train):
            if (not is_visit[index]) and (dis_tree[index] < minDis):
                minDis = dis_tree[index]
                minDis_pos = index
        
        is_visit[minDis_pos] = 1
        
        #更新最小距离
        for index in range(len_train):
            if (not is_visit[index]) and (dis_tree[index] > dis_array[index,minDis_pos]):
                dis_tree[index] = dis_array[index,minDis_pos]
                neighbour[index] = minDis_pos

    return neighbour
    










