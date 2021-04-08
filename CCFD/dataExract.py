import numpy as np
import math
import re
import csv

#从文件中提取出数据
def exractDatas(file_name):
    labelSet = []
    dataSet = []
    with open(file_name,'r') as f:
        for data_line in f.readlines():
            data = data_line.strip().split(',')
            # data = data_line.strip().split('\t')
            # for index in range(len(data)):
            #     try:
            #         data[index] = float(data[index])
            #         # data = [float(data[index]) for index in range(len(data) - 1)]
            #     except ValueError:
            #         labelSet.append(data[index])
            # dataSet.append(data)

            # one_data = []
            # # print(data)
            # for index in range(len(data)):
            #     # print(col_data)
            #     if is_number(data[index]):
            #         one_data.append(float(data[index]))
            #     else:
            #         labelSet.append(data[index])
            # dataSet.append(one_data)

            label = data[-1]
            # label = data[0]
            data = [float(data[index]) for index in range(len(data) - 1)]
            labelSet.append(label)
            dataSet.append(data)

    dataSet_mat = np.array(dataSet)
    return dataSet_mat,labelSet

def readCsv(file_name):
    labelSet = []
    dataSet = []
    reader = csv.reader(open(file_name,'r'))
    for row_data in reader:
        labelSet.append(row_data[-1])
        data = [float(row_data[index]) for index in range(len(row_data) - 1)]
        dataSet.append(data)

    dataSet_mat = np.array(dataSet)
    return dataSet_mat,labelSet




#判断是否是数字
def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False



#算出所有距离矩阵
def getAllDistance(dataSet_mat):
    line_num,col_Num = dataSet_mat.shape
    min_dis = float('inf')
    max_dis = float('-inf')
    distance_mat = np.zeros((line_num,line_num))
    for index  in range(line_num):
        for tmp in range(line_num):
            if index != tmp:
                dis = math.sqrt(np.sum(pow(dataSet_mat[index,:] - dataSet_mat[tmp,:],2)))
                if dis < min_dis:
                    min_dis = dis
                if dis > max_dis:
                    max_dis = dis
                distance_mat[index,tmp],distance_mat[tmp,index] = dis,dis
    return distance_mat,min_dis,max_dis

#计算均值
def averageData(dataSet):    
    return sum(dataSet)/len(dataSet)
    
#计算方差
def varianceData(dataSet):
    # print('data:',dataSet)
    aver = averageData(dataSet)
    temp = 0.0
    for data in dataSet:
        temp += pow(data - aver,2)
    return math.sqrt(temp/len(dataSet))

#得出我们的召回率和精确率
def getRecallAndPrecise(cluster,labelSet):
    three_class = {}
    num = 0
    for label in labelSet:
        if label not in three_class.keys():
            three_class[label] = 1
            num += 1
        else:
            three_class[label] += 1
    # print('num:',num)
    recall,precise = 0,0
    all_true_num = 0
    #簇中哪类数量最多，那么就是哪一类
    for one_cluster in cluster:
        one_cluster_three = {}
        for index in one_cluster:
            if labelSet[index] not in one_cluster_three:
                one_cluster_three[labelSet[index]] = 1
            else:
                one_cluster_three[labelSet[index]] += 1
        # print('one_cluster_three:',one_cluster_three)
        cluster_class = max(one_cluster_three,key = one_cluster_three.get)
        all_true_num += one_cluster_three[cluster_class]
        recall += one_cluster_three[cluster_class] / three_class[cluster_class]
        precise += one_cluster_three[cluster_class] / len(one_cluster)
        # print('recall:',recall)
        # print('precise',precise)
        # print('----------------------')
    recall /= num
    precise /= num
    acc = all_true_num / len(labelSet)
    return recall,precise,acc

#统计簇中几类的数量
def getThreeClassNum(labelSet):
    three_class = {}
    for label in labelSet:
        if label not in three_class.keys():
            three_class[label] = 1
        else:
            three_class[label] += 1
    return three_class


#对数据进行归一化处理
def normData(all_data,singular_point):
    data = [all_data[index] for index in singular_point]
    sum_data = sum(data)
    # print(data)
    # print('---------------------')
    # print(sum_data)
    for index in range(len(singular_point)):
        data[index] = data[index] / sum_data
    return data

