import specIndividual
import specPopulation
#遗传算法的主要步骤

def firstStage(trainDataSet,populationSize,generationNum,flod,pc,m):
    dim = len(trainDataSet[0])-1
    #算法的第一步操作，得到就是五种传统算法的使用到的维度
    bestDim = []
    for i in range(m):
        #先是初始化种群
        population = specPopulation.speciesPopulation(1,populationSize,dim)
        population.getFSPopulation()
        for j in range(generationNum):
            #先是选择
            #这里的i说明的是第几个基础分类的算法
            population.firstSelection(i,trainDataSet,flod)
            #然后变异
            population.mutation()
            #之后交叉
            population.crossover(pc)
        populationDim = population.getBest()
        bestDim.append(populationDim)
    return bestDim

def secondStage(labelSet,C,populationSize,generationNum,m,pc):
    #算法的第二步操作
    population = specPopulation.speciesPopulation(2,populationSize,m)
    population.getFSPopulation()
    for i in range(generationNum):
        #先是选择
        population.secondSelection(C,labelSet)
        #然后变异
        population.mutation()
        #之后交叉
        population.crossover(pc)
    return population.getBest()
