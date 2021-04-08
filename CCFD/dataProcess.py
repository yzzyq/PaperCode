import numpy as np
import dataExract
import math

#根据距离矩阵，计算出每个点范围内的点个数
def getDensity(distance_mat,d_threshold):
    line_num,col_num = distance_mat.shape
    all_density = np.zeros(line_num)
    for index in range(line_num):
        all_density[index] = len([distance_mat[index,col] for col in range(col_num) if distance_mat[index,col] < d_threshold]) - 1
    return all_density

#根据密度，计算出所有距离
def getAllDis(all_density,distance_mat):
    sort_index = np.argsort(-all_density)
    all_dis = np.zeros(len(all_density))
    num = 0
    # print(sort_index)
    for index in sort_index:
        if 0 == num:
            all_dis[index] = max(distance_mat[index,:])
        else:
            # print(num)
            # print([distance_mat[index,sort_index[index_min]] for index_min in range(num) if index_min != index])
            all_dis[index] = min([distance_mat[index,sort_index[index_min]] for index_min in range(num) if sort_index[index_min] != index])
        num += 1
    return all_dis

#找出我们需要的离群点
def excludedPoint(n,all_r,k):
    singular_point = []
    singular_value = []
    aver_data = dataExract.averageData(all_r)
    # print('aver_data:',aver_data)
    for index in range(len(all_r)):
        if all_r[index] <= n*aver_data:
            singular_point.append(index)
            singular_value.append(all_r[index])
    # print('singular_point:',singular_point)
    # print('singular_value:',singular_value)
    aver_data = dataExract.averageData(singular_value)
    variance_data = dataExract.varianceData(singular_value)
    len_singular = len(singular_point)
    Standard_deviation = math.sqrt((len_singular- 1)/len_singular*variance_data)

    # print(aver_data)
    # print(Standard_deviation)
    #print(aver_data - 2*Standard_deviation) 
    chioce_point = []
    # print('aver_data - 5*Standard_deviation:',aver_data - 5*Standard_deviation)
    # print('aver_data + 5*Standard_deviation:',aver_data + 5*Standard_deviation)
    for index in range(len_singular):
        # print('singular_value[index]:',singular_value[index])
        if singular_value[index] <= aver_data - 5*Standard_deviation or\
            singular_value[index] >= aver_data + 5*Standard_deviation:
            chioce_point.append(singular_point[index])
    #如果没有，那么取出最小最大的
    if len(chioce_point) < k:
        num = 0
        singular_value_sort = np.argsort(singular_value)
        while len(chioce_point) < k:
            index = singular_value_sort[num]
            n_index = singular_value_sort[-num]
            if singular_point[index] not in chioce_point and all_r[singular_point[index]] != 0:
                chioce_point.append(singular_point[index])
            if singular_point[n_index] not in chioce_point and all_r[singular_point[n_index]] != 0:
                chioce_point.append(singular_point[n_index])
            num += 1
    return chioce_point


#使用K近邻进行聚类
def getClsuter(distance_mat,center_point,cluster):
    line_num = distance_mat.shape[0]
    print(center_point)
    for index in range(line_num):
        if index not in center_point:
            center_dis = np.array([distance_mat[index,center_index] for center_index in center_point])
            # print(center_dis)
            max_index = np.argmin(center_dis)
            cluster[max_index].append(index)

#计算适应度
def getFitness1(cluster,distance_mat):
    m = len(cluster)
    sum_dis = 0
    for one_cluster in cluster:
        center_point = one_cluster[0]
        point_num = len(one_cluster)
        one_dis = 0
        for data_index in range(point_num-1):
            one_dis += distance_mat[center_point,one_cluster[data_index+1]]
        sum_dis += one_dis/point_num
    return sum_dis / m

#计算适应度
def getFitness2(cluster,distance_mat):
    m = len(cluster)
    sum_dis = 0
    if m == 1:
        return 0
    for one_cluster in range(m):
        one_cluster_dis = 0
        for two_cluster  in range(m):
            if one_cluster != two_cluster:
                one_cluster_dis += distance_mat[one_cluster,two_cluster]
        sum_dis += one_cluster_dis/(m-1)
    return sum_dis / m





