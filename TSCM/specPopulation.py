import specIndividual
import random
import copy
#种群
class speciesPopulation:

    def __init__(self,num,populationSize,genesNum):
        self.genesNum = genesNum
        self.num = num
        self.populationSize = populationSize
        self.population = []

    #对种群个体进行设置 
    def getFSPopulation(self):
        while len(self.population) < self.populationSize:
            individual = specIndividual.speciesIndividual(self.num,\
                                                          self.genesNum)
            self.population.append(individual)

    #根据适应度排序
    def sortFitness(self):
        for i in range(self.populationSize):
            for j in range(self.populationSize - i - 1):
                if self.population[j].fitness < self.population[j+1].fitness:
                    self.population[j+1],self.population[j] = \
                                    self.population[j],self.population[j+1]
            
    #选择
    def firstSelection(self,algoithmNum,trainDataSet,flod):
        train = copy.deepcopy(trainDataSet)
        #计算种群的适应度
        for i in range(self.populationSize):
            self.population[i].calndividualFitness(\
                self.num,algoithmNum,train,flod)
        
        #根据适应度进行排序
        self.sortFitness()
                
        #将最大适应度的物种复制到新种群中去。最大的占新种群的1/4
        newPopulation = []
        maxNum = self.populationSize // 4
        for i in range(maxNum):
            newPopulation.append(self.population[0])
        #其他的物种按照适应度大小进入
        for j in range(self.populationSize - maxNum):
            newPopulation.append(self.population[j+1])

        self.population = newPopulation

    #第二步的选择
    def secondSelection(self,C,labelSet):
        #计算每个个体的适应度
        for i in range(self.populationSize):
            self.population[i].calSecondFitness(C,labelSet)

        #根据适应度进行排序
        self.sortFitness()

        #将最大适应度的物种复制到新种群中去。最大的占新种群的1/4
        newPopulation = []
        maxNum = self.populationSize // 4
        for i in range(maxNum):
            newPopulation.append(self.population[0])
        #其他的物种按照适应度大小进入
        for j in range(self.populationSize - maxNum):
            newPopulation.append(self.population[j+1])

        self.population = newPopulation
        
        

    #变异
    def mutation(self):
        #第一阶段和第二阶段变异是不同的
        mutationLimit = 0.8
        for i in range(self.populationSize):
            #print('genes:',self.population[i].genes)
            rate = random.random()
            if rate > mutationLimit:
                if self.num == 1:
                    interval = random.randint(0,self.genesNum - 1)
                    if self.population[i].genes[interval] == 1:
                        self.population[i].genes[interval] = 0
                    else:
                        self.population[i].genes[interval] = 1
                elif self.num == 2:
                    for j in range(self.genesNum):
                        interval = random.uniform(-0.1,0.1)
                        self.population[i].genes[j] += interval
                        if self.population[i].genes[j] < 0:
                            self.population[i].genes[j] = 0

    #交叉
    def crossover(self,pc):
        crossProbability = random.random()
        if crossProbability > pc:
            #随机选取俩个不相同的数
            crossFirst = random.randint(0,self.populationSize-1)
            crossSecond = random.randint(0,self.populationSize-1)
            while crossFirst == crossSecond:
                crossSecond = random.randint(0,self.populationSize-1)
            self.crossGenes(0,self.genesNum // 2,crossFirst,crossSecond)
            self.crossGenes(self.genesNum // 2,self.genesNum-1,\
                            crossFirst,crossSecond)

     #基因交换
    def crossGenes(self,begin,end,crossFirst,crossSecond):
        temp1 = random.randint(begin,end)
        temp2 = random.randint(begin,end)
        while temp2 == temp1:
            temp2 = random.randint(begin,end)
        begin = min(temp1,temp2)
        end = max(temp1,temp2)
        self.population[crossFirst].genes[begin:end],\
        self.population[crossSecond].genes[begin:end] = \
        self.population[crossSecond].genes[begin:end],\
        self.population[crossFirst].genes[begin:end]


     #获得适应度最大的个体
    def getBest(self):
        maxFitness = self.population[0].fitness
        maxIndividual = 0
        for i in range(self.populationSize):
            if self.population[i].fitness > maxFitness:
                maxIndividual = i
                maxFitness = self.population[i].fitness
        return self.population[maxIndividual].genes
             
        
        
        
            
        
    
    
