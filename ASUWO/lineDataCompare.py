import selfDataCompare as sdc
import numpy as np

#选择的元素有Hb、O3、O3、S2、S2
#查看线性表中的发射线的情况
def getOtherEmissionLine(dataSet,dataProcess,side):
    all_Centers = [4862,6549,6585,6718,6732]
    i = 0
    sideLen = 2*side + 1
    #每条数据的处理
    for data in dataSet:
        all_Trust = []
        for center in all_Centers:
            trust = sdc.getOneValue(data,sideLen,center)
            # if trust > 0:
            all_Trust.append(trust)
        if len(np.where(np.array(all_Trust[1:]) > 1.2)[0]) >= 2:
            for index,trust in enumerate(all_Trust):
                if trust > 2:
                    all_Trust[index] += all_Trust[index]
                # elif trust > 1:
                #     all_Trust[index] += 1
        # if i == 421 or i == 160:
        #     for index,wave in enumerate(data[0]):
        #         if 6722 < wave < 6742:
        #             print('{0},{1}'.format(wave,data[1][index]))
        #     print('{0}:{1}'.format(i,all_Trust))

        dataProcess[i,1] = sum(all_Trust)

        if (all_Trust[1] + all_Trust[2] == 0) or (all_Trust[3] + all_Trust[4] == 0):
            if dataProcess[i,0] < 1:
                dataProcess[i,1] = 0
                dataProcess[i,-1] = -1
            # else:
            #     dataProcess[i,1] = sum(all_Trust)
        # else:
        #     all_Trust[1] += abs(all_Trust[1] - all_Trust[2])
        #     all_Trust[3] += abs(all_Trust[3] - all_Trust[4])
        #     dataProcess[i,1] = sum(all_Trust)
        # dataProcess[i,1] = sum(all_Trust)
        i += 1
