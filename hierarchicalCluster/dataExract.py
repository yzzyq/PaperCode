import matplotlib.pyplot as plt
import numpy as np

#数据的提取
def extractOneDataSet(filname):
    dataSet = []
    #读取plt文件中的数据
    i = 0
    with open(filname,'r') as f:
        for line in f.readlines():
            if i > 5:
                #文件前六行都是说明，从第六行开始读取
                lineData = line.strip().split(',')
                data = [float(lineData[j]) for j in range(len(lineData)) if j < 2]
                data.append(float(lineData[4]))
                dataSet.append(data)
            i += 1
    return dataSet

def getplot(dataSet):
    dataSetMat = np.array(dataSet)
    fig = plt.figure(figsize=(8,6))
    plt.scatter(dataSetMat[:,0],dataSetMat[:,1])
    plt.title('scatter plot')
    plt.show()
    
    
