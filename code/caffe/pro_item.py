import copy
import os
import multiprocessing

import config

def extPath(args):
    [begin, end, itemData, subDirDic] = args
    result = []
    missImgNum = 0
    for i in range(begin, end):
        tmp = itemData[i].split()
        if not tmp:
            continue
        itemName = tmp[0]+'.jpg'
        line = ''
        ex = 0
        for subDir in subDirDic:
            if itemName in subDirDic[subDir]:
                line += (subDir+'/'+itemName+' ')
                ex = 1
                break
    return (result, missImgNum)

def main():
    itemDirPath = '../data/'
    subDirList = ['tianchi_fm_img2_1', 'tianchi_fm_img2_2', 'tianchi_fm_img2_3', 'tianchi_fm_img2_4']
    itemPath = '../data/offline/test_items.txt'
    proItemPath = '../data/offline/pro_test_item.txt'
    if config.online:
        itemPath = '../data/offline/test_items.txt'
        proItemPath = '../data/offline/pro_test_item.txt'

    with open(itemPath, 'r') as itemFile:
        itemData = itemFile.readlines()
    subDirDic = {}
    for subDir in subDirList:
        subDirDic[subDir] = set([filename for filename in os.listdir(itemDirPath+subDir) if 'jpg' in filename]) 
    for subDir in subDirDic:
        print(subDir, len(subDirDic[subDir]))
    args = []
    step = len(itemData)/1
    for i in range(0, 1):
        tmp = [i*step, min((i+1)*step, len(itemData))]
        tmp += [copy.deepcopy(itemData), copy.deepcopy(subDirDic)]
        args.append(tmp)

    #pool = multiprocessing.Pool(processes = config.poolSize)
    #results = pool.map(extPath, args)
    results = []
    results.append(extPath(args[0]))
    itemFile = open(proItemPath, 'w')
    for result in results:
        print(result[1])
        for line in result[0]:
            itemFile.write(line)

if __name__=='__main__':
    main()
