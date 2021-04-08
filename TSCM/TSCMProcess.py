import mainRun
import dataExract
import time
import ordinaryClassifiers as oc
import numpy as np
import copy
import os


#基于遗传算法的俩步分类
#主程序
if not os.path.exists('Result'):
    os.mkdir('Result')
if not os.path.exists('Time'):
    os.mkdir('Time')
test_catalog = 'test'
train_catalog = 'train'
listdirs = os.listdir(trainCatalog)
for dirs in listdirs:
    print(dirs)
    train_path = train_catalog+'/'+dirs+'/'
    test_path = test_catalog+'/'+dirs+'/'
    #获取训练数据和测试数据
    train_dataSet = dataExract.extractData(train_path,dirs,'traindata.txt')
    test_dataSet = dataExract.extractData(test_path,dirs,'testdata.txt')
    #数据的初始化
    population_size = 40
    generation_num = 4
    #数据分成训练和测试的比率
    flod = 4
    #交叉操作的概率
    pc = 0.5
    m = 5
    #我们使用的是五种基础的分类器，KNN,NB,LR,DT,SVM
    #first stage:分类器的训练
    start_time = time.clock()
    print('开始选择维度。。')
    choice_dim = mainRun.firstStage(train_dataSet,population_size,\
                                   generation_num,flod,pc,m)
    #得到的是每个训练数据在根据第一步的结果在传统机器学习算法中的结果
    print('维度选择结束。。')
    C = oc.getResults(train_dataSet,choice_dim)
    train_labelSet = np.array(train_dataSet)[:,-1]
    print('开启第二步。。')
    weights = mainRun.secondStage(train_labelSet,C,population_size,\
                                  generation_num,m,pc)
    #second stage:投票
    print('权重算出来。。')
    results = []
    weights = np.array(weights)
    exe_dataSet = copy.deepcopy(test_dataSet)
    i = 0
    for data in exe_dataSet:
        print(i)
        C = oc.testOutCome(data,m,choice_dim,train_dataSet)
        one_result = np.sum(weights*C)
        if one_result <= 0:
            one_result = -1
        else:
            one_result = 1
        results.append(one_result)
        i += 1
    print('得出测试结果')
    endTime = time.clock()
    needTime = '  时间：'+ str(endTime - startTime) +'s'
    with open('Time/'+dirs+'_time.txt','w') as f:
        f.write(needTime)

    for i in range(len(results)):
        if results[i] == 1:
            with open('Result/'+dirs+'_positive_category.txt','a') as f:
                f.write(str(i)+'\n')
        else:
            with open('Result/'+dirs+'_negative_category.txt','a') as f:
                f.write(str(i)+'\n')
