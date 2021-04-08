import random
import numpy as np
import math
import random


#二元切分，可以固定树的数据结构
class treeNode:
    def __init__(self,results = None,leftBranch = None,rightBranch = None,feature = None,value=None):
        self.leftBranch = leftBranch   #左子树
        self.rightBranch = rightBranch   #右子树
        self.feature = feature  #划分的特征 
        self.value = value      #划分的特征值
        self.results = results  #划分结果
       

#对类别可能的值进行计数
def uniquecounts(dataSet):
    decisionType = {}
    for data in dataSet:
        if data[-1] not in decisionType.keys():
            decisionType[data[-1]] = 0
        decisionType[data[-1]] += 1
    return decisionType

def getMost(dataSet):
    decisionType = uniquecounts(dataSet)
    return max(decisionType,key = decisionType.get)
    


#计算连续变量的熵值
def entropy(dataSet):
    results = uniquecounts(dataSet)
    ent = 0.0
    for r in results.keys():
        ratio = float(results[r])/len(dataSet)
        ent = ent - ratio*(math.log(ratio)/math.log(2))
    return ent

#连续数值候选的集合
def continuousCandidate(continuousDataSet):
    splitSet = []
    for i in range(len(continuousDataSet) - 1):
        temp = (continuousDataSet[i] + continuousDataSet[i+1])/2
        splitSet.append(temp)
    return splitSet
        

#连续变量的划分点的选择
def chooesContinueDivision(dataSet,col):
    #得到这一列中所有的值
    data_col = [example[col] for example in dataSet]
    data_col.sort()
    #得出切分点的候选集
    splitSet = continuousCandidate(data_col)
    #最好的切分点
    best_split = 0.0
    best_gain = 1000.0
    best_sets = None
    #选取信息增益最大的那个切分点
    for split in splitSet:
        leftPoint = []
        rightPoint = []
        for i in range(len(data_col)):
            if split >= dataSet[i][col]:
                leftPoint.append(dataSet[i])
            else:
                rightPoint.append(dataSet[i])
        
        left_gain = entropy(leftPoint)
        right_gain = entropy(rightPoint)
        gain = left_gain + right_gain
        if gain < best_gain:
            best_gain = gain
            best_split = split
            best_sets = [leftPoint,rightPoint]
    return best_gain,best_split,best_sets

#删除使用过的特征
def delUsedCol(dataSet,del_col):
    relData = []
    if dataSet == None:
        return dataSet
    else:
        for data in dataSet:
            #已经用过的属性列，就抛去
            colRows = list(data[:del_col])
            colRows.extend(data[del_col+1:])
            relData.append(colRows)
        return relData

  
#构建树的过程
def buildTree(dataSet):
    decision = [example[-1] for example in dataSet]
    decisionType = set(decision)
    if len(decisionType) == 1:
        return treeNode(results = decision[0])
    
    #if len(dataSet) == 0:
        #return treeNode()
    
    #最佳切分
    score_all = entropy(dataSet)
    best_gain = 0.0
    best_attribute = None
    best_value = 0.0
    best_sets = None
    del_col = 0
    colNum = len(dataSet[0]) -1
    for col in range(colNum):
        #获取连续变量的信息增益 
        col_entropy,col_value,col_sets = chooesContinueDivision(dataSet,col)
        #得到信息增益
        attributeGain = score_all - col_entropy
        if attributeGain > best_gain:
            best_gain = attributeGain
            best_attribute = col
            del_col = col
            #最好的那个切分点
            best_value = col_value
            best_sets = col_sets
    #创建子分支 
    if best_gain > 0:
        leftBranch = buildTree(best_sets[0])
        rightBranch = buildTree(best_sets[1])
        return treeNode(leftBranch = leftBranch,rightBranch = rightBranch,
                        feature = best_attribute,value = best_value)
    else:
        return treeNode(results = getMost(dataSet))

#测试  
def classify(test,tree):
    if tree.results != None:
        return tree.results
    else:
        num = int(tree.feature)
        col = test[num]
        branch = None
        if col > tree.value:
            branch = tree.rightBranch
        else:
            branch = tree.leftBranch
        return classify(test,branch)

#打印出树
def printTree(tree,indent = ' '):
    if tree.results != None:
        print(str(tree.results))
    else:
        print(str(tree.feature)+':'+str(tree.value)+"?")

        print(indent+'小于'+ str(tree.value) + "->")
        printTree(tree.leftBranch,indent+" ")
        print(indent + '大于'+ str(tree.value)+'->')
        printTree(tree.rightBranch,indent+' ')

