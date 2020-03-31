import time
import random as rd

from tqdm import *

def main():
    dataDir = '../data/taobao/'
    dataDir = '../data/amazon/'
    
    #itemFilePath = dataDir+'match_item.txt'
    #matchPairPath = dataDir+'match_pair.txt'
    #trainSetPath = dataDir+'train_set_1to1.txt'
    #testItemPath = dataDir+'test_item_1to1.txt'
    #testGroundPath = dataDir+'test_ground_1to1.txt'
    
    itemFilePath = dataDir+'match_item_all.txt'
    matchPairPath = dataDir+'match_pair_all.txt'
    trainSetPath = dataDir+'train_set_1to1_all.txt'
    testItemPath = dataDir+'test_item_1to1_all.txt'
    testGroundPath = dataDir+'test_ground_1to1_all.txt'

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
  
    testRatio = 0.0
    NegRatio = float(len(matchPairSet))/(len(itemList)*(len(itemList)-1))*2
    
    print('Generating training and testing set...', time.ctime())
    trainSetFile = open(trainSetPath, 'w')
    testItemFile = open(testItemPath, 'w')
    testGroundFile = open(testGroundPath, 'w')
    for i in tqdm(range(0, len(itemList))):
        for j in range(i+1, len(itemList)):
            itemA = itemList[i]
            itemB = itemList[j]
            pairId = itemA+','+itemB
            pairIdR = itemB+','+itemA
            if pairId in matchPairSet or pairIdR in matchPairSet:
                if rd.random()<=testRatio:
                    testItemFile.write(itemA+'\n'+itemB+'\n')
                    testGroundFile.write(pairId+'\n'+pairIdR+'\n')
                else:
                    trainSetFile.write(pairId+',1\n'+pairIdR+',1\n')
            else:
                if rd.random()<=NegRatio:
                    trainSetFile.write(pairId+',0\n'+pairIdR+',0\n')

if __name__=='__main__':
    main()
