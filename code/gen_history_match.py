from datetime import date
import time
import math
import cPickle as pickle

def dateDiff(aList, bList):
    result = 365*10
    for a in aList:
        for b in bList:
            dateA = date(int(a[0:4]),int(a[4:6]), int(a[6:8]))
            dateB = date(int(b[0:4]),int(b[4:6]), int(b[6:8]))
            result = min(result, abs((dateA-dateB).days))
    return  result

def genMatchDic(userHistoryPath, dumpPath):
    userHistoryData = open(userHistoryPath, 'r').readlines()
    userDic = {}
    for line in userHistoryData:
        tmp = line.split()
        userId = tmp[0]
        itemId = tmp[1]
        date = tmp[2]
        if userId not in userDic:
            userDic[userId] = {}
        if itemId not in userDic[userId]:
            userDic[userId][itemId] = []
        userDic[userId][itemId].append(date)

    timeDT = 7
    matchDic = {}
    for user in userDic:
        for itemA in userDic[user]:
            for itemB in userDic[user]:
                if itemA != itemB and dateDiff(userDic[user][itemA], 
                        userDic[user][itemB]) <= timeDT:
                    pairId = itemA+','+itemB
                    if pairId not in matchDic:
                        matchDic[pairId] = 0
                    matchDic[pairId] += 1
    print('The size of matchDic is %s' % len(matchDic))
    with open(dumpPath, 'w') as f:
        pickle.dump(matchDic, f)

def loadMatchDic(dumpPath):
    return pickle.load(open(dumpPath))

def countMatch(matchDic):
    userLT = 100
    pairNum = 0
    for pairId in matchDic:
        if matchDic[pairId]>=userLT:
            pairNum += 1
    print('%s pairs are bought together by more than %s people' % (pairNum ,userLT))

def main():
    fresh = True
    userHistoryPath = '../data/user_bought_history.txt'
    dumpPath = '../data/match.dat'
    matchDic = None
    if fresh:
        print('Generating matchDic...', time.ctime())
        genMatchDic(userHistoryPath, dumpPath)
    else:
        print('Loading matchDic...', time.ctime())
        matchDic = loadMatchDic(dumpPath)
    print('Finished...', time.ctime())
    countMatch(matchDic)

if __name__=='__main__':
    main()
