#local projection for high-dimensional outlier detection
import dataExract
import localProject
import numpy as np
import os


if not os.path.exists('Result'):
    os.mkdir('Result')
if not os.path.exists('ResultRun'):
    os.mkdir('ResultRun')
catalog = 'test'
listdirs = os.listdir(catalog)
for dirs in listdirs:
    print(dirs)
    path = catalog+'/'+dirs+'/'+'testdata.txt'
    #提取数据
    dataSet,labelSet = dataExract.extractData(path,dirs)
    dataLocOuts = []
    a = 0.5
    #进行y的循环操作
    for i in range(len(dataSet)):
        print(i)
        listlocOut = localProject.process(dataSet,i,a)
        dataLocOuts.append(listlocOut)
    #对每个数据的locOut数据进行计算
    sumLocOut = np.sum(dataLocOuts,axis = 0)
    sortLocOut = np.argsort(sumLocOut)
    label = []
    for loc in sortLocOut:
        label.append(labelSet[loc])
    
    num = 0
    if dirs == 'Hbeta_OIII':
        num = 1635
    else:
        num = 1633
    trueResults = []
    falseResults = []
    for i in range(len(sortLocOut)):
        if i < num:
            trueResults.append(sortLocOut[i])
        else:
            falseResults.append(sortLocOut[i])
            
    cluster = str('正类别：') + str(trueResults) \
                  + str('    负类别：') + str(falseResults)    
    fileName = dirs + '.txt'
    
    f = open('ResultRun/'+fileName,'w')
    f.write(cluster)
    f.close()
        
    recall,precise = dataExract.computeRecall(label,dirs)
    print('得出召回率：',recall)
    print('得出精确度：',precise)
    #将得到的簇输入到文件中

    f = open('Result/'+fileName,'w')
    cluster = '召回率：' + str(recall) + '    精确度：' + str(precise)
    f.write(cluster)
    f.close()
