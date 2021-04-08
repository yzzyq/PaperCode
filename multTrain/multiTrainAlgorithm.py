import numpy as np
import KNN
from sklearn.decomposition import PCA
from sklearn import tree
import randomTreeAlgorithm
import naiveBayes
import itertools
# import Tree
import csoAlgorithm
import copy

class MultiTrainA:

    def __init__(self, feature_manipulation_num, learn_algorithm_num, vote_confident):
        self.feature_manipulation = [None]*feature_manipulation_num
        # self.learn_algorithm = np.zeros(learn_algorithm_num)
        self.learn_algorithm = [None]*learn_algorithm_num
        self.vote_confident = vote_confident
        self.N = len(self.feature_manipulation)*len(self.learn_algorithm)
        self.classifiers = [None]*self.N
        self.false_threshold = [0.5]*self.N
        self.l = np.zeros(self.N)

    def initClassifiers(self, train, label):
        #四个基础分类器分别是随机树，贝叶斯，J4.8决策树，k=5的knn
        randomTree = randomTreeAlgorithm.RandomTrees()
        self.learn_algorithm[0] = randomTree

        gnb = naiveBayes.GaussianNB()
        self.learn_algorithm[1] = gnb
        # gnb.fit(train,label)

        # decisionTree = Tree.DecisionTreeClassifier()
        decisionTree = tree.DecisionTreeClassifier()
        self.learn_algorithm[2] = decisionTree

        # 这里是KNN算法
        knn = KNN.KnnAlogrithm(5, train, label)
        self.learn_algorithm[3] = knn

        # #三个特征操控方法，PCA，CSO，全部特征
        # pca = PCA(n_components = 0.95)
        # # pca.fit(train)
        # self.feature_manipulation[0] = pca 

        # cso = csoAlgorithm.CSO(n = 3, population_size = 30, max_iter_num = 100, rate = 0.1)
        # self.feature_manipulation[1] = cso
        # # cso.process(train)

        # self.feature_manipulation[3] = len(train)

        #三个特征操控方法，PCA，CSO，全部特征
        pca = PCA(n_components = 2)
        # pca.fit(train)
        self.feature_manipulation[0] = pca 

        # cso = csoAlgorithm.CSO(n = 3, population_size = 30, max_iter_num = 100, rate = 0.1)
        self.feature_manipulation[1] = 'cso'
        # cso.process(train)

        self.feature_manipulation[2] = [1,1,1]

        index = 0
        print('开始初始化')
        for fea_mani,learn in itertools.product(self.feature_manipulation,self.learn_algorithm):
            learn_copy = copy.deepcopy(learn)
            if fea_mani == 'cso':
                fea_mani = csoAlgorithm.CSO(learn_fit=learn_copy)
            print('开始的算法和特征选择:',index)    
            self.classifiers[index] = [fea_mani, learn_copy]
            index += 1

        self.trainClassifier(train, label)
   
    #得到数据进行训练
    def trainClassifier(self, trainSet, labelSet):
        
        for index,fea_learn in enumerate(self.classifiers):
            train = copy.deepcopy(trainSet)
            label = copy.deepcopy(labelSet)
            print('得到数据进行训练:',index)
            fea_mani = fea_learn[0]
            learn = fea_learn[1]
            if fea_mani == [1,1,1]:
                #使用全部的特征
                # print('全部特征')
                pre_process_data = train
                # print('全部特征')
            else:
                if hasattr(fea_mani,'is_cso'):
                    # print('cso算法')
                    pre_process_data,best_col_num = fea_mani.searchBestCol(train,copy.deepcopy(label))
                    fea_mani = best_col_num
                    # print('cso算法')
                else:
                    # print('pca')
                    pre_process_data = fea_mani.fit(train).transform(train)
                    # print('pca')
            # print('维度:',len(pre_process_data[0]))
            # print('个数：',len(pre_process_data))
            # print('类别个数：',len(label))
            #有放回的采样
            # print('有放回的采样')
            train_data,train_label = self.getDataByBooststrap(pre_process_data, label)
            # print('训练算法')
            # print('个数：',len(train_data))
            # print('类别个数：',len(train_label))
            learn.fit(train_data,train_label)
            
            #既要存储特征操控方法，也要存储学习算法
            self.classifiers[index] = [fea_mani, learn]
            

    #有放回采样
    def getDataByBooststrap(self, pre_process_data, label):
        dataBooststrap_index = np.random.randint(0,len(label),size = (len(label)//2))
        train_data = [pre_process_data[index] for index in dataBooststrap_index]
        train_label = [label[index] for index in dataBooststrap_index]
        return train_data,train_label

    def predictionAlgotithm(self, train,index_algorithm):
        #vote是类别投票数，prob是置信度
        len_data = len(train)
        vote = np.zeros((len_data,2))
        prob = np.zeros((len_data,2))
        all_label = np.zeros(len_data)
        all_pro = np.zeros(len_data)
        for index,fea_learn in enumerate(self.classifiers):
            if index != index_algorithm:
                fea_mani = fea_learn[0]
                learn = fea_learn[1]
                train_copy = copy.deepcopy(train)
                # print('fea_mani:',fea_mani)
                # print('fea_learn:',fea_learn)
                if isinstance(fea_mani,list):
                    pre_process_data = self.getProcess(fea_mani, train_copy)
                else:
                    pre_process_data = fea_mani.fit(train_copy).transform(train_copy)
                # if fea_mani == [1,1,1]:
                #     #使用全部的特征
                #     pre_process_data = train_copy
                # else:

                #     if hasattr(fea_mani,'is_cso'):
                #         pre_process_data = fea_mani.searchBestCol(train_copy)
                #     else:
                #         pre_process_data = fea_mani.fit(train_copy).transform(train_copy)
                    # pre_process_data = fea_mani.fit(copy.deepcopy(train))
                all_label = []
                all_confidence = []
                if hasattr(learn,'is_write'):
                    all_label,all_confidence = learn.predict(pre_process_data)
                else:
                    all_label = learn.predict(pre_process_data)
                    temp_con = learn.predict_proba(pre_process_data)
                    for one_con in temp_con:
                        all_confidence.append(float(max(one_con)))
                for index,label in enumerate(all_label):
                    if label == -1:
                        vote[index,0] += 1
                        prob[index,0] += all_confidence[index]
                    else:
                        vote[index,label] += 1
                        prob[index,label] += all_confidence[index]
        
        for line_index in range(len_data):
            data_class = np.argmax(vote[line_index,:])
            all_pro[line_index] = prob[line_index,data_class]
            if data_class == 0:
                data_class = -1
            all_label[line_index] = data_class
        return all_label,all_pro
 
        # data_class = np.argmax(vote)
        # data_pro = prob[data_class]
        # if data_class == 0:
        #     data_class = -1
        # return data_class,data_pro

    #计算错误率
    def calcFalseRatio(self, train_data, train_label, index_algorithm):
        sum_false = 0
        all_label,all_pro = self.predictionAlgotithm(train_data,index_algorithm)
        false_sum = sum([train_label[index] != all_label[index] for index in range(len(train_label))]) 
        return sum_false/len(train_data)

        # for index,data in enumerate(train_data):
        #     data_class,data_pro = self.predictionAlgotithm(data, index_algorithm)
        #     if data_class != train_label[index]:
        #         sum_false += 1
        # return sum_false/len(train_data)
   
    #根据列，得到我们想要的数据
    def getProcess(self, best_col_num, trainSet):
        trains = np.array(trainSet)
        for index,col in enumerate(best_col_num):
            if col == 0:
                np.delete(trains, index, axis=1)
        return trains


    #根据rank进行采样
    def subsampling(self,add_data,add_label,out_num,rank):
        rank_index = np.argsort(rank)
        add_index = rank_index[out_num:]
        new_add_data = []
        new_add_label = []
        for index in add_index:
            new_add_data.append(add_data[index])
            new_add_label.append(add_label[index])
        return new_add_data,new_add_label

