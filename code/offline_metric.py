import math

def metric(groundFilePath, resultFilePath, upper_bound):
    groundDic = {}
    with open(groundFilePath, 'r') as groundFile:
        groundData = groundFile.readlines()
    for ground in groundData:
        tmp = ground.split()
        itemId = tmp[0]
        itemList = tmp[1].split(',')
        groundDic[itemId] = set(itemList)    
    with open(resultFilePath, 'r') as resultFile:
        resultData = resultFile.readlines()
    apList = []
    hitSum = 0
    noResult = 0
    groundSum = 0
    itemSet = set()
    notFull = 0
    predictSum = 0    
    for result in resultData:
        hitNum = 0
        ap = 0
        tmp = result.split()
        itemId = tmp[0]
        if itemId not in groundDic:
            apList.append(0)            
            continue
        if len(tmp) < 2:
            noResult += 1
            continue
        if itemId in itemSet:
            continue
        itemSet.add(itemId)
        itemList = tmp[1].split(',')
        upper = min(upper_bound, len(itemList))
        if upper < 200:
            notFull += 1
        predictSum += upper        
        for i in range(0, upper):
            item = itemList[i]
            delta = 0
            if item in groundDic[itemId]:
                hitNum += 1
                delta = 1
                hitSum += 1
            if hitNum:
                ap += (1.0/(1-math.log(hitNum/(i+1.0), math.e)))*delta
        apList.append(ap/len(groundDic[itemId]))
        groundSum += len(groundDic[itemId])
    print(upper_bound, len(apList), len(groundDic))
    score = sum(apList)/len(groundDic)
    print('map@200: %s' % score)
    print predictSum, hitSum, groundSum, noResult, notFull

def main():
    resultFilePath = '../data/offline/fm_submissions_topic.txt'
    groundFilePath = '../data/offline/ground.txt'
    metric(groundFilePath, resultFilePath, 100)
    #for i in range(0,201,10):
    #    metric(groundFilePath, resultFilePath, i)
    #    print('-'*50)
        

if __name__ == '__main__':
    main()
