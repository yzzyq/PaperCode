import random
import numpy as np

#得出第一个值
def getOneValue(data,sideLen,center):
    peak_value = []
    before_slope = None
    isMout = False
    mout_value = []
    len_data = len(data[0])
    value = 0
    min_value = float('inf')
    trust = 0
    # slope_value = 0
    if data[0][-1] > center+(sideLen-1)//2:
        for i in range(len_data):
            if center-(sideLen-1)//2 <= data[0][i] <= center+(sideLen-1)//2:
                if min_value > data[1][i]:
                    min_value = data[1][i]
                value = data[1][i+1]
                #需要记录峰值
                slope = (data[1][i+1] - data[1][i])/(data[0][i+1] - data[0][i])
                #距离边缘值
                if before_slope == None:
                    peak_value.append(data[1][i])
                else:
                    if before_slope > 0 and slope < 0:
                        #如果是6564附近的值，需要记录下来
                        if center-2 < data[0][i] < center+2:
                            #分为好几种情况
                            if abs(before_slope) > 0.02 and abs(slope) > 0.02:
                                if center-2 < data[0][i] < center - 1:
                                    isMout = True
                                    mout_value.append(data[1][i])
                                elif center - 1 < data[0][i] < center + 2:
                                    isMout = True
                                    mout_value.append(data[1][i])
                            # slope_value = before_slope
                        else:
                            peak_value.append(data[1][i])
                    elif before_slope < 0 and slope > 0:
                        if center < data[0][i] < center + 1:
                            isMout = False
                            break
                        # if center + 1 < data[0][i] < center + 2 and abs(abs(before_slope) - abs(slope)) < 1:
                        #     isMout = False
                        #     break

                before_slope = slope
                #检查是否有峰值，如果没有可以直接退出来
                if data[0][i] >= center+2 and not isMout:
                    break
        if min_value > value:
            min_value = value 
        peak_value.append(value)
        peak_value = np.array(peak_value)
##        print(mout_value)
##        print(peak_value)
##
##        print('--------------------------')
        if isMout:
            # tmp = (max(mout_value) - min_value) / (peak_value - min_value)
            # trust = sum(tmp) / len(tmp) 
            trust = (max(mout_value) - min_value) / (max(peak_value) - min_value)
        # print('mount_value:{0},min_value:{1},max(peak_value):{2}'.format(mout_value,min_value,max(peak_value)))       
    return trust


#dataProcess就是进行组合的数据
def getFluxData(dataSet,side,center,dataProcess):
    sideLen = 2*side + 1
    lenData = len(dataSet)
    label = np.zeros(len(dataSet))
    for i in range(lenData):
        trust = getOneValue(dataSet[i],sideLen,center)
        if trust > 0:
            if trust > 2:
                label[i] = 1
        else:
            #这个数据没有发射线
            label[i] = -1
        dataProcess[i,0] = trust
    return label
