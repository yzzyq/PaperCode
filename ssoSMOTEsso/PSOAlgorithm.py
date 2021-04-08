from sklearn.model_selection import KFold
import Tree
import numpy as np
from sklearn import tree
# 粒子群优化算法


class PSO:

    def __init__(self, indicate_num,
                       inertia_weight = 0.689,
                       cognititve_coefficient = 1.43,
                       social_coefficient = 1.43,
                       init_population = 5,
                       max_iter_num = 100):    
        self.inertia_weight = inertia_weight
        self.cognititve_coefficient = cognititve_coefficient
        self.social_coefficient = social_coefficient
        self.init_population = init_population
        self.r1 = np.random.rand()
        self.r2 = np.random.rand()
        self.indicate_num = indicate_num
        self.max_iter_num = max_iter_num

        #X是位置，Y是速度
        self.X = np.zeros((init_population, indicate_num))
        self.Y = np.zeros((init_population, indicate_num))
        
        #个体的最佳适应度以及它的表现
        self.each_best = np.zeros((init_population, indicate_num))
        self.each_fitness = np.zeros(init_population) 

        #群体最佳适应度以及它的表现    
        self.global_best = np.zeros(indicate_num)
        self.global_fitness = float('inf')


    def initPopulation(self,maj_train, maj_label, min_train, min_label):
        self.X = np.random.rand(self.init_population, self.indicate_num)
        self.Y = np.random.rand(self.init_population, self.indicate_num)

        #开始的时候肯定就是最佳适应度
        self.each_best = self.X
        
        for individual_index in range(self.init_population):
            print('计算适应度:',individual_index)
            one_fitness = self.getFitness(self.X[individual_index], maj_train, maj_label, min_train, min_label)
            self.each_fitness[individual_index] = one_fitness
            if one_fitness < self.global_fitness:
                self.global_fitness = one_fitness
                self.global_best = self.X[individual_index]


    def getFitness(self, X, maj_train, maj_label, min_train, min_label):
        #先通过X找出训练数据
        choose_train, choose_label = self.chooseDataByX(X, maj_train, maj_label, min_train, min_label)
        ten_data = KFold(n_splits = 10)
        error_rate = 0
        for train_index,test_index in ten_data.split(choose_train):
            train_data = [choose_train[index] for index in train_index]
            train_label = [choose_label[index] for index in train_index]

            test_data = [choose_train[index] for index in test_index]
            test_label = [choose_label[index] for index in test_index]

            decision_Tree = tree.DecisionTreeClassifier()
            decision_Tree.fit(train_data,train_label)
            # print('len(test_data):',len(test_data))
            test_result = decision_Tree.predict(test_data)
            # print('test_result:',test_result)
            error_rate += sum([test_result[index] != test_label[index] for index in range(len(test_result))]) / len(test_result)
        return error_rate / 10

    def chooseDataByX(self, X, maj_train, maj_label, min_train, min_label):
        choose_train = [maj_train[pos_index] for pos_index in range(len(X)) if X[pos_index] > 0.5]
        choose_label = [maj_label[pos_index] for pos_index in range(len(X)) if X[pos_index] > 0.5]
        choose_train.extend(min_train)
        choose_label.extend(min_label)
        return choose_train,choose_label
    

    def processSearch(self, maj_train, maj_label, min_train, min_label):
        #初始化群体
        # print('初始化群体')
        self.initPopulation(maj_train, maj_label, min_train, min_label)
        
        current_iter = 0
        while current_iter < self.max_iter_num:
            # print('迭代的次数:',current_iter)
            for index in range(self.init_population):
                one_fitness = self.getFitness(self.X[index],maj_train, maj_label, min_train, min_label)
                print('self.each_fitness[index]:',self.each_fitness)
                if one_fitness < self.each_fitness[index]:
                    self.each_fitness[index] = one_fitness
                    self.each_best[index] = self.X[index]
                    if one_fitness < self.global_fitness:
                        self.global_fitness = one_fitness
                        self.global_best = self.X[index]
            #更新位置和速度
            for index in range(self.init_population):
                self.Y[index] = self.inertia_weight*self.Y[index] + \
                                self.cognititve_coefficient*self.r1*(self.each_best[index] - self.X[index]) + \
                                self.social_coefficient*self.r2*(self.global_best - self.X[index])
                self.X[index] = self.X[index] + self.Y[index]
            current_iter += 1
        choose_train = [maj_train[pos_index] for pos_index in range(self.indicate_num) if self.global_best[pos_index] > 0.5]
        choose_label = [maj_label[pos_index] for pos_index in range(self.indicate_num) if self.global_best[pos_index] > 0.5]
        return choose_train,choose_label




