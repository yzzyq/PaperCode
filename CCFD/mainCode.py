#CCFD
import dataExract 
import numpy as np
import dataProcess
import time
import copy



#进行初始化
# file_name = 'Iris（有标签）.txt'
# file_name = 'wdbc（有标签）.txt'
# file_name = 'heart（整齐标签有标签）.txt'
# file_name = 'sonar(有标签).txt'
file_name = 'waveform40.csv'
# file_name = 'train.csv'
# file_name = 'Wall-Following.csv'
# file_name = 'Frogs_MFCCs.csv'
result_file_name = file_name.split('.')[0] + '-result.txt'
d_threshold = 0
#去除点时需要的系数
n = 2
#判断中心点时需要的系数
k = 3
#百分比左右调整的幅度
r = 3
#开始中心的百分比
percent = 15

#从文件中取出我们需要的数据
# dataSet_mat,labelSet = dataExract.exractDatas(file_name)
dataSet_mat,labelSet = dataExract.readCsv(file_name)

print(len(dataSet_mat))
print(len(labelSet))
print(set(labelSet))

print('算出所有距离矩阵')
#算出所有距离矩阵
distance_mat,min_dis,max_dis = dataExract.getAllDistance(dataSet_mat)
print('min_dis:',min_dis)
print('max_dis:',max_dis)
# print(distance_mat)

# init_dis = min_dis + (max_dis - min_dis)*percent

#最好的效果的threshold
best_threshold = 0
best_fitness = 0
best_cluster = []
best_center_point = []

num = 1
while 0 < r:
    # print('迭代的次数：',num)
    fitnesses = []
    clusters = []
    center_points = []
    thresholds = [] 
    for i  in range(2):
        if i == 0:
            d_threshold = min_dis + (max_dis - min_dis)*((percent - r)/100)
        if i == 1:
            d_threshold = min_dis + (max_dis - min_dis)*((percent + r)/100)
        thresholds.append(d_threshold)
        # print('d_threshold:',d_threshold)
        # print('计算all_density')
        #根据距离矩阵，计算出每个点范围内的点个数
        print('d_threshold:',d_threshold)
        all_density = dataProcess.getDensity(distance_mat,d_threshold)
        print('all_density:',all_density)
        # print('计算all_dis')
        #根据密度，计算出所有距离
        all_dis = dataProcess.getAllDis(all_density,distance_mat)
        print('all_dis:',all_dis)

        all_r = all_density*all_dis

        # print('计算all_r')
        # print(all_r)

        singular_point = dataProcess.excludedPoint(n,all_r,k)

        all_dis_tmp = dataExract.normData(all_dis,singular_point)
        all_density_tmp = dataExract.normData(all_density,singular_point)
        # print('singular_point:',singular_point)
        # print(all_dis_tmp,all_density_tmp)

        #根据离群点的条件找出符合条件的中心点
        center_point = []
        cluster = []
        for s_index in range(len(singular_point)):
            if all_dis_tmp[s_index]!= 0 and all_density_tmp[s_index] != 0:
                # print(all_dis_tmp[s_index]/all_density_tmp[s_index],all_density_tmp[s_index]/all_dis_tmp[s_index])
                if all_dis_tmp[s_index]/all_density_tmp[s_index] < k and all_density_tmp[s_index]/all_dis_tmp[s_index] < k:
                    center_point.append(singular_point[s_index])
                    center_tmp = [singular_point[s_index]]
                    cluster.append(center_tmp)
        # print('cluster:',cluster)
        # print('center_point:',center_point)

        if len(center_point) > k:
            center_point_copy = copy.deepcopy(center_point)
            tmp_r = np.array([all_r[index] for index in center_point_copy])
            # tmp_r = np.array([all_dis_tmp[point_index]/all_density_tmp[point_index] + all_density_tmp[point_index]/all_dis_tmp[point_index] \
                # for point_index in range(len(center_point_copy)) if all_density_tmp[point_index] != 0 and all_dis_tmp[point_index] != 0])
            sort_r = np.argsort(-tmp_r)
            center_point = []
            cluster = []
            for index in range(k):
                center_point.append(center_point_copy[sort_r[index]])
                center_tmp = [center_point_copy[sort_r[index]]]
                cluster.append(center_tmp)

        center_points.append(center_point)
                
            # for index in range(k):
            #     center_tmp = []
            #     center_tmp.append(singular_point[index])
            #     center_point.append(singular_point[index])
            #     cluster.append(center_tmp)
            
        start_time = time.clock()
        # print('得出所有的中心点：',cluster)

        #根据中心点利用K近邻算法找出簇
        dataProcess.getClsuter(distance_mat,center_point,cluster)
        clusters.append(cluster)
        # print('簇：',cluster)
        #计算出他们的适应度
        fitness1 = dataProcess.getFitness1(cluster,distance_mat)
        fitness2 = dataProcess.getFitness2(center_point,distance_mat)
        fitness =  fitness2/fitness1
        fitnesses.append(fitness)
    fitnesses = np.array(fitnesses)
    max_index = np.argmax(fitnesses)
    if max_index == 0:
        percent -= r
    if max_index == 1:
        percent += r
    r -= 0.5
    if fitnesses[max_index] > best_fitness:
        best_fitness = fitnesses[max_index]
        best_cluster = clusters[max_index]
        best_center_point = center_points[max_index]
        best_threshold = thresholds[max_index] 
    num += 1
    print(best_center_point)

print('选出最优的')    
print('最优的适应度:',best_fitness)
print('最好的簇:',best_cluster)
print('最好的中心点:',best_center_point)
print('最好的阈值：',best_threshold)

end_time = time.clock()
#得出我们的召回率和精确率
recall,precise,acc = dataExract.getRecallAndPrecise(best_cluster,labelSet)
print('得到召回率和精确率')
print('recall:',recall)
print('precise:',precise)
print('acc:',acc)
run_time = str(end_time - start_time) + 's'

#将结果写入到文件中去
with open(result_file_name,'w') as f:
    result = '时间：' + run_time + '，' + '召回率：' + str(recall) + '，' + '精确率：' + str(precise) + '，准确性：' + str(acc)
    f.write(result)

