#visit point cluster
import visitPoint

def vpCluster(VP,dataSet,eps):
    #首先得到所有VP的质心
    centroids = getAllCentroids(VP,dataSet)
    lenCent = len(centroids)
    #访问点的聚类结果
    CS = []
    for i in range(lenCent):
        #得到该点的所有邻近的点
        N = neighbour(i,centroids,eps)
        #print('neighbour:',N)
        for i in range(len(CS)-1,-1,-1):
            if is_joinable(N,CS[i]):
                N = list(set(N + CS[i]))
                del CS[i]
        if len(N) > 2:
            CS.append(N)
    return CS

#首先得到所有VP的质心
def getAllCentroids(VP,dataSet):
    allCentroids = []
    for cluster in VP:
        centroid = visitPoint.getCentroid(cluster,dataSet)
        allCentroids.append(centroid)
    return allCentroids

#得到该点的所有邻近的点
def neighbour(point,centroids,eps):
    allNeighbour = []
    length = len(centroids)
    for i in range(length):
        if i != point and visitPoint.distance(centroids[i],centroids[point]) < eps:
            allNeighbour.append(i)
    return allNeighbour

#查看俩个簇是否有相同的元素
def is_joinable(N,cluster):
    for data in N:
        if data in cluster:
            return True
    return False
