from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os
from sklearn import tree
# import graphviz
# import pydotplus
# from IPython.display import Image
# os.environ["PATH"] += os.pathsep +'C:/Program Files(x86)/Graphviz2.38/bin/'

np.set_printoptions(threshold=np.inf)
# 根据索引得到我们所需要的训练和测试数据
def getDataByIndex(data, label, train_index, test_index):
    train_data = [data[index] for index in train_index]
    train_label = [label[index] for index in train_index]
    test_data = [data[index] for index in test_index]
    test_label = [label[index] for index in test_index]
    return train_data,train_label,test_data,test_label

# 级联中一个等级是有三层的
class ForestLevel:
    def __init__(self, num_estimator, num_forests, level_index, one_level_layer_num, classification_num, max_depth, n_fold):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.level_index = level_index
        self.one_level_layer_num = one_level_layer_num
        self.classification_num = classification_num
        self.max_depth = max_depth
        self.n_fold = n_fold
        self.level_model = []

    def train(self, train_data, train_label):
        #K折交叉验证
        error_num = 0
        # train_data_num = 0
        print('--------------------largethreelayer-----------')
        print('一个等级中层数:', self.one_level_layer_num)
        train_result = []
        for layer_index in range(self.one_level_layer_num):
            print('layer' + str(layer_index))
            if layer_index != 0:
                train_data[layer_index] = np.concatenate((train_data[layer_index],train_result), axis = 1)
            # 开始一层的训练            
            one_layer_train_data = train_data[layer_index]
            one_layer_train_label = train_label[layer_index]
            print('使用的是哪层数据：', layer_index)
            print('数据量是:',len(one_layer_train_data),len(one_layer_train_label))
            print('数据的特征数是:',len(one_layer_train_data[0]))
            kf = KFold(5, True, self.n_fold).split(one_layer_train_data)
            # 对级联中一层进行训练
            one_layer = ForestLayer(self.num_forests, self.num_estimator, layer_index, self.max_depth, 1)
            all_validation_data = []
            all_validation_label = [] 
            # 交叉验证
            
            train_result = np.zeros((len(one_layer_train_data), self.classification_num*self.num_forests))
            
            for train_index, test_index in kf:
                one_train_data, one_train_label, one_test_data, one_test_label = getDataByIndex(one_layer_train_data, 
                                                                                                one_layer_train_label, 
                                                                                                train_index, 
                                                                                                test_index)

                one_train_result = one_layer.train(one_train_data, one_train_label)
               
                # print('one_train_result',one_train_result) 
                for o_index,index in enumerate(train_index):
                    train_result[index, :] += one_train_result[o_index,:]
                    # print('train_result',train_result)
                # 记录好这一层的验证集
                all_validation_data.extend(one_test_data)
                # print('all_validation_data',all_validation_data)
                all_validation_label.extend(one_test_label)
                # print('all_validation_label',all_validation_label)
            self.level_model.append(one_layer)
            all_three_test_result = one_layer.one_layer_model[0].predict_proba(all_validation_data)
            
            # print('all_three_test_result第一个', all_three_test_result)
            for model_index in range(1,len(one_layer.one_layer_model)):
                # print('model_index',model_index)
                tmp = one_layer.one_layer_model[model_index].predict_proba(all_validation_data)
                # print('tmp', tmp)
                all_three_test_result += tmp
                # print('all_three_test_result第二个:',all_three_test_result)
            all_test_result = np.argmax(all_three_test_result, axis = 1)
            # print(' all_test_result第三个', all_test_result)
            
            # 计算这一层的误差,只能用于这个数据之中
            error_num += one_layer.compute_error_rate(all_test_result, train_label)
            # print('all_test_result',all_test_result)
            # print('train_label',train_label)
            # 数据增加特征
            train_result = train_result / (self.n_fold - 1)
            # print('train_result', train_result)
            pd.set_option('display.max_columns',1000)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth',1000)

            print('数据增加的特征：',train_result)
            print('共有的数据条数：',len(train_result))
        print('一共有：', len(self.level_model))
        print('---------largethreelayer  ending--------')
        # print('error_num', error_num)
        return error_num / (len(train_label)*len(train_label[0])), train_result
    
    
    def predict(self, test_data):
        all_result = []
        # print('test_data:', test_data)
        for layer_index in range(self.one_level_layer_num):
            if layer_index != 0:
                test_data[layer_index] = np.concatenate((test_data[layer_index],all_result), axis = 1)
            one_level = self.level_model[layer_index]
            one_layer_test_data = test_data[layer_index]
            all_result = one_level.predict(one_layer_test_data)
            # all_result.append(one_class)
        return all_result


# 级联森林中的一层
class ForestLayer:
    def __init__(self, num_forests, n_estimators, layer_index, max_depth, min_samples_leaf):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.layer_index = layer_index
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.one_layer_model = []

    def train(self, train_data, train_label):
        # 对具体的layer内的森林进行构建
        # val_prob = np.zeros((len(set(one_layer_train_label))*self.num_forests,len(one_layer_train_data)))
        val_prob = np.zeros((len(train_data),len(set(train_label))*self.num_forests))
        # print('一层中森林的个数:', self.num_forests)
        for forest_index in range(self.num_forests):
            # print(forest_index)
            #前俩个是随机森林
            if forest_index <2:
                #子树的个数
                clf = RandomForestClassifier(n_estimators= self.n_estimators,
                                             n_jobs= -1,
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label)
                #####可视化##########
                #dot_data = tree.export_graphviz(clf, out_file=None,
                         #feature_names=train_data,
                         #class_names=train_label,
                         #filled=True, rounded=True,
                         #special_characters=True)
                #graph = pydotplus.graph_from_dot_data(dot_data)
                #Image(filename= 'tree.png')
                #########################可视化################
                #记录类向量
                # print(clf.predict_proba(val_data[:25,:]).T)
                val_prob[:, forest_index*2:forest_index*2+2] = clf.predict_proba(train_data)
                    
                # print('val_prob', val_prob[forest_index*2:forest_index*2+2, :])
                #组建layer层
                if len(self.one_layer_model) < self.num_forests:
                    self.one_layer_model.append(clf)
                else:
                    self.one_layer_model[forest_index] = clf
    
            else:
                # 后俩个是极端随机森林
                clf = ExtraTreesClassifier(n_estimators= self.n_estimators,
                                             n_jobs= -1,
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label)
                val_prob[:, forest_index*2:forest_index*2+2] = clf.predict_proba(train_data)
                
                # print('val_proa', val_prob[forest_index*2:forest_index*2+2, :])                
                # print('val_proa', val_prob[forest_index*2:forest_index*2+2, :])
                 
                #组建layer层
                # self.one_layer_model.append(clf)
                if len(self.one_layer_model) < self.num_forests:
                    self.one_layer_model.append(clf)
                else:
                    self.one_layer_model[forest_index] = clf
        return val_prob
 
    # 错误率的计算
    def compute_error_rate(self, result, label):
        B = ['F','G']        
        # result = [one_result + 1 for one_result in result]
        pre_result = []
        for one_result in result:
            pre_result.append(B[one_result])
            # result[index] = B[one_result] 
           
        error_data = [pre_result[index] != label[index] for index in range(len(label))]
        # print('error_data',error_data)         
        error_num = error_data.count(1)

        return error_num

    def predict(self, one_layer_test_data):
        # print('one_layer_test_data:', one_layer_test_data)
        one_layer_result = self.one_layer_model[0].predict_proba(one_layer_test_data)
        # print('len(one_layer_result):', len(one_layer_result))
        # print('len(one_layer_result[0]):', len(one_layer_result[0]))
        print(len(self.one_layer_model))
        for model_index in range(1,len(self.one_layer_model)):
            tmp = self.one_layer_model[model_index].predict_proba(one_layer_test_data)
            one_layer_result = np.concatenate((one_layer_result,tmp), axis = 1)
        # print('len(one_layer_result):', len(one_layer_result))
        # print('len(one_layer_result[0]):', len(one_layer_result[0]))
        return one_layer_result 



















#定义层类
# class Layer:
#     def __init__(self, n_estimators, num_forests,  max_depth = 30, min_samples_leaf = 1):
#         #定义森林数
#         self.num_forests = num_forests
#         #定义每个森林的树个数
#         self.n_estimators = n_estimators
#         #每棵树的最大深度
#         self.max_depth = max_depth
#         #树会生长到所有叶子都分到一个类，或者某节点所代表的样本数已小于min_sample_leaf
#         self.min_samples_leaf = min_samples_leaf
#         #最后产生的类向量
#         self.model = []


#     #训练函数
#     def train(self, train_data, train_label, val_data):
#         print('-------------------------调用森林模型-------------------')
#         # train_data, test_data, train_label, test_label = train_test_split(train_data,train_label)
#         #定义出该层的类向量，有self.num_forest行，val_data.shape[0]列，val_data就是weight
#         val_prob = np.zeros([self.num_forests*2, val_data.shape[0]])

#         #对具体的layer内的森林进行构建
#         for forest_index in range(self.num_forests):
            
            
#             #如果是第偶数个设为随机森林
#             if forest_index % 2 == 0:
#                 #子树的个数
#                 clf = RandomForestClassifier(n_estimators= self.n_estimators,
#                                             n_jobs= -1,
#                                             max_depth=self.max_depth,
#                                             min_samples_leaf=self.min_samples_leaf)
                
#                 clf.fit(train_data, train_label)
#                 #记录类向量
#                 # print(clf.predict_proba(val_data[:25,:]).T)
#                 val_prob[forest_index*2:forest_index*2+2, :] = clf.predict_proba(val_data).T
                
                                
#                 # print('val_prob', val_prob[forest_index*2:forest_index*2+2, :])
#                 #组建layer层
#                 self.model.append(clf)
                
#             #如果是第奇数个就设为极端森林
#             else :
#                 clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
#                                           n_jobs= -1,
#                                           max_depth=self.max_depth,
#                                           min_samples_leaf=self.min_samples_leaf)
#                 clf.fit(train_data, train_label)
#                 val_prob[forest_index*2:forest_index*2+2, :] = clf.predict_proba(val_data).T
#                 # print('val_proa', val_prob[forest_index*2:forest_index*2+2, :])                
#                 # print('val_proa', val_prob[forest_index*2:forest_index*2+2, :])
                 
#                 #组建layer层
#                 self.model.append(clf)
           
#             # print('val_prob:', val_prob)
#             # print(len(val_prob))        
#         #对每一类求平均值            
#         val_avg_mean = np.sum(val_prob, axis = 1) / len(val_data)
#         print(len(val_avg_mean))
#         print('val_avg_mean', val_avg_mean)
#         # val_concatenate_mean = val_avg_mean.T
#         # print(len(val_concatenate_mean))
#         # print('val_concatenate_mean ', val_concatenate_mean)
#         #返回平均结果和转置后的类向量矩阵
#         return val_avg_mean
#         print('---------------------------------调用结束-----------------------------')
#     #定义预测函数，也是最后一层的功能
#     def predict(self, test_data):
#         print('------------------------------森林测试--------------------------')
#         classifiction_num = 2
#         predict_result = np.zeros([self.num_forests*2, test_data.shape[0]])
#         print('predict_result:', predict_result)
#         for forest_index, clf in enumerate(self.model):
#             predict_result[forest_index*2:forest_index*2+2, :] = clf.predict_proba(test_data).T
#         predict_result_concatenate = predict_result.transpose((1,0))
#         print('predict_result', predict_result[forest_index*2:forest_index*2+2, :])
#         print(len( predict_result))
#         print('森林测试结束')

#         # all_data_result = np.zeros([classifiction_num,test_data.shape[0]])
        
#         # for one_classification in range(classifiction_num):
#             # for one_model_result in range(len(self.model)):
#                 # all_data_result[one_classification,:] = predict_result[one_model_result,one_classification]
#             # # all_data_result[one_classification,:] /= len(self.model)
#             # print('每一类的平均值：', all_data_result[one_classification,:])
#         # max_data = np.max(all_data_result, axis = 0)
#         # print('max_data:', max_data)
       
#         # all_data_result_concatenate = all_data_result.transpose((1,0))
#         return [predict_result,predict_result_concatenate]


# class KfoldWarpper:
#     #定义森林中使用树的个树，k折的个数，k-折交叉验证，第几层， 最大深度， 最小叶子节点限制的数
#     def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, max_depth = 100, min_samples_leaf = 1):
#         self.num_forests = num_forests
#         self.n_estimators = n_estimators
#         self.n_fold = n_fold
#         self.kf = kf
#         self.layer_index = layer_index
#         self.max_depth = max_depth
#         self.min_samples_leaf = min_samples_leaf
#         self.model = []
#     #用k交叉验证进行训练
#     def train(self, train_data, train_label):
#         print('--------------------------Layer开始执行-------------------------------')
#         num_samples, num_features = train_data.shape
#         print('num_samples:', num_samples)
#         print('num_features', num_features)

#         val_prob = np.empty([num_samples])
#         print('v',len(val_prob))
#         # 创建新的空矩阵，num_smples行，numforest列，用于放置预测结果(把4个森林训练的结果进行连接，属于层级部分)
#         val_prob_concatenate = np.empty([num_samples, self.num_forests])
        
#         #进行k折交叉验证，在train_data里创建交叉验证的补充
#         for train_index, test_index in self.kf:
#             print(train_index,test_index)
#             # print('train_label:',train_label)
#             # print('train_data:',train_data)
#             #选出训练集
#             X_train = train_data[train_index, :]
#             # print('X_train:', X_train)
#             print(len(X_train))
#             #验证集
#             X_val = train_data[test_index, :]
#             # print('X_val:', X_val)
#             print(len(X_val))
#             #训练标签
#             y_train = np.array(train_label)[train_index]
#             # print('y_train:', y_train)
#             print(len(y_train))
#             print('--------------------不知道干啥--------------------')
#             #调用层类
#             layer = Layer(self.n_estimators, self.num_forests, self.max_depth, self.min_samples_leaf)
#             #调用上面的Layer类里面的train对森林进行模型训练，记录输出的结果
#             print('-----------------------------Layer结束---------------------')
#             temp_one = layer.train(X_train, y_train, X_val)
            
#             #在模型中填充层级
#             self.model.append(layer)
#         return [val_prob, val_prob_concatenate]
        
#     #定义预测函数，树模型交叉验证用作下一层的训练
#     def predict(self, test_data):
#         print('-------------------------开始训练模型-------------------------------')
#         test_prob = np.zeros([8,test_data.shape[0]])
#         test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests*2])
#         # print('test_prob_concatenate :',test_prob_concatenate )
#         for layer in self.model:
#             temp_prob, temp_prob_concatenate = \
#                 layer.predict(test_data)
#             test_prob += temp_prob
#             temp_prob_concatenate += temp_prob_concatenate
            
#         test_prob /=self.n_fold
#         test_prob_concatenate /= self.n_fold
#         print('test_prob', test_prob)
#         print('----------------------------结束测试-------------------')
#         return [test_prob, test_prob_concatenate]    
            
