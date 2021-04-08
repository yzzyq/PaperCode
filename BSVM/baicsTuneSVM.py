from sklearn.model_selection import KFold
from sklearn import svm
import numpy as np
import metricsMethod


class btSVM:

    def __init__(self, C, gamam, r_threshold):
        self.C = C
        self.gamam = gamam
        self.r_threshold = r_threshold
        # self.try_data = try_data
    
    def getWeight(self, train, label):
        bSVM = svm.SVC(self.C, 'rbf', gamma = self.gamam)
        # bSVM = svm.SVC(self.C, 'linear')
        bSVM.fit(train, label)
        #------获取权重
        # print(rbfKenrnel(train,train[index]))
        support_vector =[metricsMethod.rbfKenrnel(train,train[index],self.gamam) for index in bSVM.support_]
        # print('support_vector:',support_vector)
        # print('支持向量的个数：',len(support_vector))
        # print('支持向量的维数：',len(support_vector[0]))
        weights = bSVM.dual_coef_[0].dot(support_vector)
        # print('weights的维数:',len(weights))
        # w = bSVM.coef_[0]
        # a = - w[0]/w[1]
        # return a
        return weights

    def getAlphasAndBeta(self,weights,train,label):
        pos_index = [index for index,data in enumerate(label) if data == 1]
        neg_index = [index for index,data in enumerate(label) if data == -1]
        #计算其中最小beta,beta是正类数据
        # print(rbfKenrnel(train, train[Index]))
        # temp_weights = metricsMethod.rbfKenrnel(train, train[Index])
        # print([np.dot(weights.T,metricsMethod.rbfKenrnel(train, train[index],self.gamam)) for index in pos_index])
        Z_pos = sorted([np.dot(weights,metricsMethod.rbfKenrnel(train, train[index],self.gamam)) for index in pos_index])
        # Z_pos = sorted([np.dot(weights,train[index]) for index in pos_index])
        beta = Z_pos[0]
        #计算其中最大alphas，alphas是负类数据
        # temp_weights = metricsMethod.rbfKenrnel(train, train[Index])
        alphas = max([np.dot(weights,metricsMethod.rbfKenrnel(train, train[index],self.gamam)) for index in pos_index])
        # alphas = max([np.dot(weights,metricsMethod.rbfKenrnel(train, train[index],self.gamam)) for index in pos_index])
        return beta,alphas,Z_pos

    def getBias(self,beta,alphas,Z_pos):
        if alphas < beta:
            bias = (1 - self.r_threshold)*beta + self.r_threshold*alphas
        else:
            bias = np.percentile(Z_pos,1-self.r_threshold)
        return bias    

    # def rbfKenrnel(self,x,y):
    #     K = np.zeros(len(x))
    #     for j in range(len(x)):
    #         delta = x[j,:] - y
    #         k[j] = delta*delta.T
    #     K = np.exp(K/(-1*self.gamam**2))
    #     return K






    

    
