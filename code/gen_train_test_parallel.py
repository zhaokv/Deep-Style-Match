import time
import random as rd

from tqdm import *

def select(args):
    [begin, end, itemList, matchPairSet, NegRatio] = args
    result = []
    for i in tqdm(range(begin, end)):
        for j in range(i+1, len(itemList)):
            itemA = itemList[i]
            itemB = itemList[j]
            pairId = itemA+','+itemB
            pairIdR = itemB+','+itemA
            if pairId in matchPairSet or pairIdR in matchPairSet:
                result.append(pairId+',1\n'+pairIdR+',1\n')
            else:
                if rd.random()<=NegRatio:
                    result.append(pairId+',0\n'+pairIdR+',0\n')
    return result

def main():
    dataDir = '../data/taobao/'
    dataDir = '../data/amazon/'
    
    itemFilePath = dataDir+'match_item_all.txt'
    matchPairPath = dataDir+'match_pair_all.txt'
    trainSetPath = dataDir+'train_set_1to1_all.txt'

    print('Loading items into list...', time.ctime())
    with open(itemFilePath, 'r') as itemFile:
        itemData = itemFile.readlines()
    itemList = []
    for item in itemData:
        tmp = item.split()
        itemList.append(tmp[0])
    
    print('Loading match pairs into set...', time.ctime())
    with open(matchPairPath, 'r') as matchPairFile:
        matchPairData = matchPairFile.readlines()
    matchPairSet = set()
    for line in matchPairData:
        matchPair = line.split()[0]
        tmp = matchPair.split(',')
        pairIdR = tmp[1]+','+tmp[0]
        if pairIdR not in matchPairSet:
            matchPairSet.add(matchPair)
  
    NegRatio = float(len(matchPairSet))/(len(itemList)*(len(itemList)-1))*2
    
    print('Generating training and testing set...', time.ctime())
    args = []
    step = len(itemList)/config.poolSize
    for i in range(0, config.poolSize):
        tmp = [i*step, min((i+1)*step, len(itemList))]
        tmp += [itemList, matchSet, ratio]
        args.append(tmp)

    pool = multiprocessing.Pool(processes = config.poolSize)
    results = pool.map(select, args)
    trainSetFile = open(trainSetPath, 'w')
    for result in results:
        for line in result:
            trainSetFile.write(line)

if __name__=='__main__':
    main()
