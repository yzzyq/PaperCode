import random
import numpy as np
import randomTreeAlgorithm
import naiveBayes
import Tree
import KNN
from sklearn.model_selection import KFold
from sklearn import tree


class CSO:
    
    def __init__(self, n = 3, population_size = 30, max_iter_num = 5, 
                 influence = 0.1, discrete_threshhold = 0.5,learn_fit = None):
        self.n = n
        self.population_size = population_size
        self.max_iter_num = max_iter_num
        self.influence = influence
        self.discrete_threshhold = discrete_threshhold
        self.R1 = random.random()
        self.R2 = random.random()
        self.R3 = random.random()
        self.learn_fit = learn_fit
        self.is_cso = 1


    #降维操作
    def searchBestCol(self, trainSet, labelSet):
        #初始化粒子的位置和速度
        all_position = np.random.rand(self.population_size,len(trainSet[0]))
        all_velocity = np.random.rand(self.population_size,len(trainSet[0]))
        # all_fea_selection = {}

        iter_num = 0
        choice_fea = []
        choice_fitness = []
        while iter_num < self.max_iter_num:
            # choice_fea = []
            # choice_fitness = []
            aver_all_pos = np.array([np.mean(all_position[:,one]) for one in range(len(trainSet[0]))])
            for particle_index in range(self.population_size):
                choose_col = self.continueToDiscrete(all_position[particle_index])
                # print('choose_col:',choose_col)
                #只要有一列选中，那么就可以做
                if any(choose_col):
                    # if list(choose_col) not in all_fea_selection.values():
                    if list(choose_col) not in choice_fea:
                        #计算适应程度,通过n交叉验证来计算适应度
                        choice_fea.append(list(choose_col))
                        fitness = self.getFitness(trainSet,choose_col,labelSet)
                        # print('fitness:',fitness)
                        # all_fea_selection[fitness] = list(choose_col)
                        choice_fitness.append(fitness)
            
            #记录是否已经更新
            p_update = [index for index in range(self.population_size)]
            #更新粒子
            while len(p_update) > 1:
                # choose_two_data = random.sample(p_update,2)
                # for index in choose_two_data:
                #     del p_update[index]
                # # [del p_update[index] for index in choose_two_data]
                choose_two_data = []
                two_data_col = []
                while len(choose_two_data) < 2 and len(p_update) > 0:
                    choose_one = random.randint(0,len(p_update)-1)
                    del p_update[choose_one]
                    temp = self.continueToDiscrete(all_position[choose_one])
                    if any(temp):
                        two_data_col.append(temp)
                        choose_two_data.append(choose_one)
                
                if len(choose_two_data) < 2:
                    break

                # two_data_col = [self.continueToDiscrete(all_position[index]) for index in choose_two_data]
                two_fitness = []
                for one_col in two_data_col:
                    # temp_result = filter(lambda x:one_col == x[1] , all_fea_selection.items)
                    # two_fitness.append(list(temp_result)[0][0])
                    # print('one_col:',one_col)
                    # for key,value in all_fea_selection.items():
                        # print('value:',value)
                        # if list(one_col) == list(value):
                        #     two_fitness.append(key)
                    for index in range(len(choice_fea)):
                        if list(choice_fea[index]) == list(one_col):
                            two_fitness.append(choice_fitness[index])
                # print('two_fitness:',two_fitness)
                max_index = np.argsort(-np.array(two_fitness))
                xl_index = choose_two_data[max_index[0]]
                xw_index = choose_two_data[max_index[1]]
                all_velocity[xl_index] = self.R1*all_velocity[xl_index] + \
                                         self.R2*(all_position[xw_index] - all_position[xl_index]) + \
                                         self.influence*self.R3*(aver_all_pos - all_position[xl_index])
                all_position[xl_index] += all_velocity[xl_index] 

            iter_num += 1
        best_col = np.argmax(np.array(choice_fitness))
        best_col_num = choice_fea[best_col]
        # best_col_num = all_fea_selection[max(all_fea_selection.keys())]
        pro_trainSet = self.getProcess(best_col_num, trainSet)
        return pro_trainSet,list(best_col_num)
        
    #根据阈值将连续值变成离散值
    def continueToDiscrete(self, position):
        # choose_col = np.zeros(len(trainSet[0]))
        choose_col = np.zeros(len(position))
        for col_index,col in enumerate(position):
            if col > self.discrete_threshhold:
                choose_col[col_index] = 1
        return choose_col
    
    #根据列，得到我们想要的数据
    def getProcess(self, best_col_num, trainSet):
        trains = np.array(trainSet)
        for index,col in enumerate(best_col_num):
            if col == 0:
                np.delete(trains, index, axis=1)
        return trains


    #得到适应度，这里的就是错误率
    def getFitness(self, trainSet, choose_col, label):
        train_data = self.getProcess(choose_col, trainSet)
        ten_data = KFold(n_splits = self.n)
        error_sum = 0
        for train_index,test_index in ten_data.split(train_data):
            #得出数据
            train = np.array([train_data[index] for index in train_index])
            one_train_label = np.array([label[index] for index in train_index])
            test = np.array([train_data[index] for index in test_index])
            test_label = np.array([label[index] for index in test_index])
            
            self.learn_fit.fit(train, one_train_label)
            result_label = self.learn_fit.predict(test)
            # print('len(test):',len(test))
            # print('len(test_label):',len(test_label))
            # print('len(result_label):',len(result_label))
            error_sum += np.sum([result_label[index] != test_label[index] for index in range(len(result_label))])/len(result_label)
        error_sum = error_sum / self.n
        return error_sum

    


        


