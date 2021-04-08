import numpy as np
from sklearn.ensemble import RandomForestClassifier


class VoteEnsemble:

    def __init__(self, en_size, train_len):
        #集成算法的大小
        self.en_size = en_size
        #训练样本的容量
        self.train_len = train_len
        #各样本的更新值
        self.t_x = np.zeros(train_len)
        #各样本的权重
        self.weights = np.array([1/train_len for _ in range(train_len)])
        #基础分类器
        # self.base_classifier = np.zeros(en_size)
        self.base_classifier = [None]*en_size

    #训练算法
    def trainEnsemble(self, train, label, classifier_index):
        #权重采样方法进行采样
        # print(self.weights)

        # score_samples = pow(np.random.random(),1/self.weights)
        
        score_samples = self.weights

        sample_indexs = np.argsort(-score_samples)[:self.train_len//2]
        
        #采样随机树的方式
        one_random_tree = RandomForestClassifier(n_estimators = 5)
        one_random_tree.fit([train[index] for index in sample_indexs],
                            [label[index] for index in sample_indexs])
        self.base_classifier[classifier_index] = one_random_tree
        train_result = one_random_tree.predict(train)
        return train_result
        

