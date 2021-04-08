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
                data = [lineData[i] for i in range(len(lineData)) if i < 2]
                dataSet.append(data)
    return dataSet

