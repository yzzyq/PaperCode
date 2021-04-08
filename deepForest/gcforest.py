from sklearn.model_selection import KFold
from largeThreeLayer import *
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score


# 根据索引得到我们所需要的训练和测试数据
# def getDataByIndex(data, label, train_index, test_index):
#     train_data = [data[index] for index in train_index]
#     train_label = [label[index] for index in train_index]
#     test_data = [data[index] for index in test_index]
#     test_label = [label[index] for index in test_index]
#     return train_data,train_label,test_data,test_label

#定义forest模型
class gcforest:
    def __init__(self, num_estimator, num_forests, classification_num, one_level_layer_num, max_level = 10, max_depth = 100, n_fold = 5):
        # 此级联森林中每一层森林的个数和森林中树的个数
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        # 最大层数
        self.max_level = max_level
        self.model = []
        self.classification_num = classification_num
        self.one_level_layer_num = one_level_layer_num
    
    # 训练级联森林模型
    def train(self, train_data, train_label):
        print('--------------------------------gcforest开始执行------------------------')
        
        # train_data_new = train_data.copy()
    
        # 记录前一层的误差
        before_train_loss = float('inf')
        print('before_train_loss',before_train_loss)
        # 级联森林的等级数
        level_index = 0
        
        #提前一个进行训练
        before_iter_train_data = train_data[0]
        # print('before_iter_train_data',before_iter_train_data)
        before_iter_train_label = train_label[0]
        kf = KFold(5, True, self.n_fold).split(before_iter_train_data)
        before_iter_layer = ForestLayer(self.num_forests, self.num_estimator, 0, self.max_depth, 1)
        
        train_result = np.zeros((len(before_iter_train_data), self.classification_num*self.num_forests))

        for train_index, test_index in kf:
            
            one_train_data, one_train_label, one_test_data, one_test_label = getDataByIndex(before_iter_train_data, 
                                                                                            before_iter_train_label, 
                                                                                            train_index, 
                                                                                            test_index)
            one_train_result = before_iter_layer.train(one_train_data, one_train_label)
            for o_index,index in enumerate(train_index):
                train_result[index, :] += one_train_result[o_index,:]
        
        self.model.append(before_iter_layer)
        # 数据增加特征
        train_result = train_result / (self.n_fold - 1)
        train_data[0] = np.concatenate((before_iter_train_data,train_result), axis = 1)
        print('--------gcforest---------')
        print('train_result',train_result)

        #训练级联森林中的每个等级
        while level_index < self.max_level:
            print('当前等级层数是:',level_index)
            one_level = ForestLevel(self.num_estimator, self.num_forests, level_index, self.one_level_layer_num, 
                                    self.classification_num, self.max_depth, self.n_fold)
            validation_error_rate, train_result_iter = one_level.train(train_data, train_label)
            train_data[0] = np.concatenate((train_data[0],train_result_iter), axis = 1)
            level_index += 1
            self.model.append(one_level)
            print('之前的损失是:', before_train_loss)
            print('如今的损失是:', validation_error_rate)
            if before_train_loss < validation_error_rate:
                break
            
            before_train_loss = validation_error_rate

        
             
    def predict(self, test_data):
        test_data_new = test_data.copy()
        test_prob = np.zeros((len(test_data[0]), self.classification_num))
        predict_value = np.zeros((len(test_data[0]),self.classification_num*self.num_forests))
        
        # 先对数据进行处理
        before_model = self.model[0]
        before_result = before_model.predict(test_data[0])
        test_data_new[0] = np.concatenate((test_data_new[0],before_result), axis = 1)
        
        print('第一步的处理完成') 

        for level_index in range(1, len(self.model)):
            print('level_index:', level_index)
            level = self.model[level_index]
            predict_value = level.predict(test_data_new)
            print(len(predict_value),len(predict_value[0]),len(predict_value[0]))
            test_data_new[0] = np.concatenate((test_data_new[0],predict_value), axis = 1)
            # for one_test in range(len(test_data)):
            #     print('len(test_data_new[one_test]):', len(test_data_new[one_test][0]))
            #     test_data_new[one_test] = np.concatenate((test_data_new[one_test], predict_value[one_test]), axis = 1)
            #     print('len(test_data_new[one_test]):', len(test_data_new[one_test][0]))
            # test_data_new = np.concatenate((test_data_new,predict_value),axis = 1)
        predict_value = np.array(predict_value)
        print('predict_value',predict_value)
        for forest_index in range(self.num_forests):
            print(test_prob[:,:])
            print(predict_value[:, forest_index*self.classification_num:(forest_index + 1)*self.classification_num])
            test_prob[:,:] += predict_value[:, forest_index*self.classification_num:(forest_index + 1)*self.classification_num]
        result = np.argmax(test_prob, axis=1)
            
        # gbm_score=accuracy_score(result,y_gbm_pred)
        # result = [r+1 for r in result]
        print('result',result)
        print(len(result))
        return result

    
    
            