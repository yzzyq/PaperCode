from astropy.io import fits
import os
import copy
import numpy as np
import random

#对单个光谱数据的处理
def readfits(path,fileName):
    dfu = fits.open(path + '/'+fileName)
    #初始波长
    beginWave = dfu[0].header['COEFF0']
    #步长
    step = dfu[0].header['CD1_1']
    #读出数据中的位置
    #位置
    ra = dfu[0].header['ra']
    dec = dfu[0].header['DEC']
    snrr_stn = dfu[0].header['snrr']
    snri_stn = dfu[0].header['snri']
    poistion = [float(ra),float(dec)]
    #光谱中的流量 
    flux = dfu[0].data[0]
    #求出波长,求出与流量对应的波长
    wave = np.array([10**(beginWave + step*j) for j in range(len(flux))])
    data = [wave,flux]
    #-------------------------------------------
    return data,poistion

#数据文件中的光谱数据
def exractData(fileName):
    #-------------红移-----------------
    # redshift = getRedshift(redshift_file)
    #---------------------------------
    #os.chdir(fileName)
    listFile = os.listdir(fileName)
    dataSet = []
    allPoistion = []
    for file in listFile:
        dfu = fits.open(fileName + '/'+file)
        # data_class = dfu[0].header['class']
        # print('数据的类别是：',data_class)
        # red = 0
        # if data_class == 'GALAXY':
            # red = searchRedshift(file,redshift,fileName)
        #print(file)
        #读出数据并且保存
        data,poistion = readfits(fileName,file)
        dataSet.append(data)
        allPoistion.append(poistion)
    #os.chdir(os.pardir)
    return dataSet,np.array(allPoistion)

