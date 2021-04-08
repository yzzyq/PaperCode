import numpy as np
import copy
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


class Asuwo:

    def __init__(self, c_threshold, NS, NN, k):
        self.c_threshold = c_threshold
        self.NS_nearest = NS
        self.NN_nearest = NN
        self.k_partitions = k


    def getNearest(self, maj_train, min_train):
        all_data = copy.deepcopy(min_train)
        all_data.extend(maj_train)
        len_all_data = len(all_data)
        all_point_nearest = np.zeros((len_all_data, len_all_data))
        for index,train in enumerate(all_data):
            all_point_nearest[index,index] = float('inf')
            for two_num in range(index+1, len_all_data):
                dis = np.sqrt(np.sum(pow(train - all_data[two_num], 2)))
                all_point_nearest[index, two_num], all_point_nearest[two_num, index]= dis, dis
        return all_point_nearest 
    
   
    def getDataRemoveNoise(self, all_point_nearest, min_num):
        noise_index = []
        for index,data in enumerate(all_point_nearest):
            sort_data = np.argsort(data)[0:self.NS_nearest + 1]
            is_noise = [point_index < min_num for point_index in sort_data]
            if len(set(is_noise)) > 1 and len(set(is_noise[1:])) == 1:
                noise_index.append(index)
        return noise_index

    def getBaiscData(self, one_cluster, min_train):
        add_data = []
        for data_index in one_cluster:
            add_data.append(min_train[data_index])
        return add_data     

                
    #得到我们需要的TH
    def getTH(self, min_num, one_cluster, all_point_nearest):
        sum_value = 0
        for data_index in one_cluster:
            min_value = np.min(all_point_nearest[data_index, min_num:])
            # print('min_value:',min_value)
            sum_value = 1 / ((min_value / 3) + 2)
            # if one_sum_value > 1:
            #     one_sum_value = 1
            # sum_value += one_sum_value 
        return sum_value
    
    def getError(self, one_result, test_label):
        error_num = 0
        for index in range(len(one_result)):
            if one_result[index] != test_label[index]:
                error_num += 1
        return error_num/len(one_result)

    def splitData(self, all_cluster):
        split_data = [[] for _ in range(len(all_cluster))]
        for index,one_cluster in enumerate(all_cluster):
            np.random.shuffle(one_cluster)
            one_step = len(one_cluster) // self.k_partitions
            start = 0
            for k_index in range(self.k_partitions):
                if k_index == self.k_partitions - 1:
                    split_data[index].append(one_cluster[start:])
                else:
                    # print(one_cluster[start:start+one_step])
                    split_data[index].append(one_cluster[start:start+one_step])
                start += one_step
        return split_data

    def syntheticData(self, all_cluster_size, all_cluster, min_train, min_num, maj_train,all_point_nearest):
        #首先得出我们需要的权重
        all_weights = []
        for one_cluster_index in range(len(all_cluster_size)):
            TH_data = self.getTH(min_num, all_cluster[one_cluster_index],all_point_nearest)
            one_cluster_weights = []
            for one_data_index in all_cluster[one_cluster_index]:
                sort_index = np.argsort(all_point_nearest[one_data_index, min_num:])
                TH_data = 1 / (all_point_nearest[min_num,sort_index[0]] / 3)
                F = 0
                for sort_data_index in sort_index:
                    temp = all_point_nearest[min_num,sort_data_index] / 3
                    temp = 1/temp
                    if temp <= TH_data:
                        F += temp
                    else:
                        F += TH_data
                one_cluster_weights.append(F)
            sum_weights = sum(one_cluster_weights)
            one_cluster_weights = [weight / sum_weights for weight in one_cluster_weights]
            all_weights.append(one_cluster_weights)
        #根据权重生成我们需要的数据
        over_data = []
        over_label = []
        for one_cluster_index in range(len(all_cluster_size)):
            baisc_data = self.getBaiscData(all_cluster[one_cluster_index],min_train)
            one_cluster_weights = all_weights[one_cluster_index]
            while len(baisc_data) < all_cluster_size[one_cluster_index]:
                score = np.array([pow(np.random.rand(),1/weight) for weight in one_cluster_weights])
                choose_index = np.argmax(score)
                nn_nearest = np.argsort(all_point_nearest[choose_index,:min_num])
                b = np.random.choice(nn_nearest)
                beta = np.random.rand()
                c = beta*min_train[choose_index] + (1 - beta)*min_train[b]
                baisc_data.append(c)
            over_data.extend(baisc_data)
            over_label.extend([1]*all_cluster_size[one_cluster_index])
        over_data.extend(maj_train)
        over_label.extend([-1]*len(maj_train))
        return over_data,over_label
        


    def adaptiveSubClusterSize(self, all_cluster, min_train, min_label, maj_train, maj_label):
        split_data = self.splitData(all_cluster)
        all_error = np.zeros(len(all_cluster))
        for index in range(self.k_partitions):
            train_data = copy.deepcopy(maj_train)
            train_label = copy.deepcopy(maj_label)
            test_data = []
            test_label = []
            for cluster_num in range(len(all_cluster)):
                one_cluster_test_data = []
                one_cluster_test_label = []
                for one_random in range(self.k_partitions):
                    if one_random == index:
                        [one_cluster_test_data.append(min_train[data_index]) for data_index in split_data[cluster_num][one_random]]
                        [one_cluster_test_label.append(min_label[data_index]) for data_index in split_data[cluster_num][one_random]]
                    else:
                        [train_data.append(min_train[data_index]) for data_index in split_data[cluster_num][one_random]]
                        [train_label.append(min_label[data_index]) for data_index in split_data[cluster_num][one_random]]
                test_data.append(one_cluster_test_data)
                test_label.append(one_cluster_test_label)
            lr_model = LogisticRegression(solver='liblinear')
            lr_model.fit(train_data,train_label)
            # all_error = []
            for cluster_index,cluster_data in enumerate(test_data):
                one_result = lr_model.predict(cluster_data)
                error_rato = self.getError(one_result,test_label[cluster_index])
                all_error[cluster_index] += error_rato
        all_error = [one_error / self.k_partitions for one_error in all_error]
        sum_error = sum(all_error)
        all_error = [one_error / sum_error for one_error in all_error]
        all_cluster_size = [int(one_error*len(maj_train)) for one_error in all_error]
        return all_cluster_size

    def updateAllNearest(self, all_point_nearest, min_index, cluster_num, all_cluster, min_train):
        center_point_index = all_cluster[min_index]
        for index in range(cluster_num):
            if index != min_index:
                one_dis = 0
                for point_index in center_point_index:
                    one_dis += np.sqrt(np.sum(pow(min_train[index] - min_train[point_index], 2)))
                all_point_nearest[index,min_index],all_point_nearest[min_index,index] = one_dis,one_dis
            else:
                all_point_nearest[index,min_index],all_point_nearest[min_index,index] = float('inf'),float('inf')


    def semiUnsuperviedClustering(self, min_train, min_num_one, maj_train, maj_num, all_point_nearest_one):
        # d_avg = np.mean(all_point_nearest[:min_num,:min_num])
        all_point_nearest = copy.deepcopy(all_point_nearest_one)
        min_num = min_num_one
        d_sum = np.sum([sum(all_point_nearest[index,0:index]) + sum(all_point_nearest[index,index+1:]) for index in range(min_num)])
        d_avg = d_sum / (min_num*min_num - min_num)
        # print('avg:',d_avg)
        T = d_avg*self.c_threshold
        # print('T:',T)
        #开始的时候，每个少数类都是一个簇
        all_cluster = [[index] for index in range(min_num)]
        # print('max:',self.maxDis(all_point_nearest,min_num))
        # while self.isPrcoess(all_point_nearest,min_num,T):
        # print(np.min(all_point_nearest[:min_num,:min_num]))
        min_dis = np.min(all_point_nearest[:min_num,:min_num])
        while min_dis < T:
            # min_dis = np.min(all_point_nearest)
            # print('all_point_nearest:',all_point_nearest[:min_num,:min_num])
            min_dis = np.min(all_point_nearest[:min_num,:min_num])
            # temp = np.where(all_point_nearest == np.min(all_point_nearest[:min_num,:min_num]))
            temp = np.where(all_point_nearest == min_dis)
            min_a = temp[0][0]
            min_b = temp[1][0]
            # print(min_a,min_b)
            # print('min_dis:',min_dis)
            # for index in range(maj_num):
            #     if all_point_nearest[min_a,min_num + index] < min_dis and all_point_nearest[min_b,min_num + index] < min_dis:
            #         print('min_num + index:',min_num + index)
            #         print('all_point_nearest[min_a,min_num + index]:',all_point_nearest[min_a,min_num + index])
            #         print('all_point_nearest[min_b,min_num + index]:',all_point_nearest[min_b,min_num + index])
            # mm = mm
            if any(all_point_nearest[min_a,min_num + index] < min_dis and 
                   all_point_nearest[min_b,min_num + index] < min_dis 
                   for index in range(maj_num)):
                all_point_nearest[min_a,min_b],all_point_nearest[min_b,min_a] = float('inf'),float('inf')
                # print('将之放大')
            else:
                # print('合并')
                #合并簇，就会少一个簇
                min_num -= 1
                min_index = min(min_a, min_b)
                max_index = max(min_a, min_b)
                all_cluster[min_index].extend(all_cluster[max_index])
                del all_cluster[max_index]
                np.delete(all_point_nearest, max_index, axis = 0)
                np.delete(all_point_nearest, max_index, axis = 1)
                #更新距离矩阵
                self.updateAllNearest(all_point_nearest,min_index,min_num,all_cluster,min_train)

        if any(len(one_cluster) < 3 for one_cluster in all_cluster):
            all_cluster = [[index] for index in range(min_num)]
            min_num = min_num_one
            all_point_nearest = copy.deepcopy(all_point_nearest_one)
            while any(len(one_cluster) < 3 for one_cluster in all_cluster) and min_num > 1:
                # print('all_cluster:',all_cluster)
                min_dis = np.min(all_point_nearest[:min_num,:min_num])
                temp = np.where(all_point_nearest == min_dis)
                min_a = temp[0][0]
                min_b = temp[1][0]
                print(min_a,min_b)
                min_num -= 1
                min_index = min(min_a, min_b)
                max_index = max(min_a, min_b)
                all_cluster[min_index].extend(all_cluster[max_index])
                del all_cluster[max_index]
                np.delete(all_point_nearest, max_index, axis = 0)
                np.delete(all_point_nearest, max_index, axis = 1)
                #更新距离矩阵
                self.updateAllNearest(all_point_nearest,min_index,min_num,all_cluster,min_train)
        return all_cluster

    # def isPrcoess(self,all_point_nearest,min_num,T):
        # max_dis = -float('inf')
        # is_pro = True
        # for index in range(min_num):
        #     one_max_dis = [num for num in all_point_nearest[index,:] if num != float('inf')]
        #     if max(one_max_dis) > max_dis:
        #         max_dis = max(one_max_dis) 
        # # print('max_dis:',max_dis)
        # if max_dis > T:

        # return max_dis




    def asuwoProcess(self, maj_train, maj_label, min_train, min_label):
        maj_num = len(maj_train)
        min_num = len(min_train)
        all_point_nearest = self.getNearest(maj_train, min_train)
        # print('all_point_nearest:',all_point_nearest)
        noise_index = self.getDataRemoveNoise(all_point_nearest, min_num)
        #去除噪声点
        for index in noise_index:
            np.delete(all_point_nearest, index, axis = 1)
            np.delete(all_point_nearest, index, axis = 0)
            if index < min_num:
                np.delete(min_train, index, axis = 0)
            else:
                np.delete(maj_train, index - min_num, axis = 0)
        maj_num = len(maj_train)
        min_num = len(min_train)
        #进行聚类
        all_cluster = self.semiUnsuperviedClustering(min_train, min_num, maj_train, maj_num, copy.deepcopy(all_point_nearest))
        # print('all_cluster:',all_cluster)
        #求出每个簇过采样数量
        all_cluster_size = self.adaptiveSubClusterSize(all_cluster, min_train, min_label, maj_train, maj_label)
        #合成数据
        over_data,over_label = self.syntheticData(all_cluster_size,all_cluster,min_train ,min_num,maj_train,all_point_nearest)
        return over_data,over_label





