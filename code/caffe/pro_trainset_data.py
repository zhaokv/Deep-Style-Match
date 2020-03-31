import copy
import os
import random
import multiprocessing

import config

def extPath(args):
    [begin, end, trainSetData, subDirDic] = args
    result = []
    missImgNum = 0
    for i in range(begin, end):
        tmp = trainSetData[i].split(',')
        if not tmp:
            continue
        itemName1 = tmp[0]+'.jpg'
        itemName2 = tmp[1]+'.jpg'
        label = tmp[2]
        line = ''
        ex1 = 0
        ex2 = 0
        for subDir in subDirDic:
            if itemName1 in subDirDic[subDir]:
                line += (subDir+'/'+itemName1+' ')
                ex1 = 1
                break
        for subDir in subDirDic:
            if itemName2 in subDirDic[subDir]:
                line += (subDir+'/'+itemName2+' ')
                ex2 = 1
                break
        if ex1 == 0 or ex2 == 0:
            missImgNum += 1
        else:
            line += (label)
            result.append(line)
    return (result, missImgNum)

def main():
    itemDirPath = '../../data/taobao/'
    subDirList = ['tianchi_fm_img1_1', 'tianchi_fm_img1_2', 'tianchi_fm_img1_3', 'tianchi_fm_img1_4']
    trainSetPath = itemDirPath+'train_set_1to1.txt'
    proTrainSetPath = itemDirPath+'pro_train_set_1to1.txt'

    with open(trainSetPath, 'r') as trainSetFile:
        trainSetData = trainSetFile.readlines()
    subDirDic = {}
    for subDir in subDirList:
        subDirDic[subDir] = set([filename for filename in os.listdir(itemDirPath+subDir) if 'jpg' in filename]) 
    for subDir in subDirDic:
        print(subDir, len(subDirDic[subDir]))
    args = []
    step = len(trainSetData)/1
    for i in range(0, 1):
        tmp = [i*step, min((i+1)*step, len(trainSetData))]
        tmp += [copy.deepcopy(trainSetData), copy.deepcopy(subDirDic)]
        args.append(tmp)

    #pool = multiprocessing.Pool(processes = config.poolSize)
    #results = pool.map(extPath, args)
    results = []
    results.append(extPath(args[0]))
    proTrainSetFile = open(proTrainSetPath, 'w')
    for result in results:
        print(result[1])
        for line in result[0]:
            proTrainSetFile.write(line)

if __name__=='__main__':
    main()
