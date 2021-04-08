import itertools
import random
import numpy as np

class SMOTE:
    
    def __init__(self, k):
        self.k = k

    def oversample(self, min_train, min_label):
        all_new_data = []
        for index,bench_data in enumerate(min_train):
            all_dis = []
            for index_two in range(index+1,len(min_train)):
                dis = np.sqrt(np.sum(pow(bench_data - min_train[index_two],2)))
                all_dis.append(dis)
            # for compared_data in min_train:
            #     dis = np.sqrt(np.sum(pow(bench_data - compared_data,2)))
            #     all_dis.append(dis)
            k_index = np.argsort(all_dis)[1:self.k+1]
            for index in k_index:
                new_data = bench_data + random.random()*(min_train[index] - bench_data)
                all_new_data.append(new_data)
        all_new_data.extend(min_train)
        all_new_label = np.ones(len(all_new_data))
        return all_new_data,all_new_label   




