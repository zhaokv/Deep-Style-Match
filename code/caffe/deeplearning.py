#coding:utf-8
import config
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import heapq

 
def getFeatures(rawdata, itemFeature):
    """rawdata为对应的匹配对(itema, itemb)
    
    """      
    trainXa = []
    trainXb = []
    for item in rawdata:
        a = item[0]
        b = item[1]        
        trainXa.append(itemFeature[a])
        trainXb.append(itemFeature[b])
    trainX = np.abs(trainXa - trainXb)
    trainX = trainX.T
    return trainX

def predict(arg):
    
    [testList, itemCategoryDic,
                     dicpath, trainModel, itemCorpusDic] = arg        
    if len(testList)==0:
        return []
    
    print "predict begin"
    
    result = []
    for i in testList:        
        cate = itemCategoryDic[i]
        testData=[(i, k) for k in itemCategoryDic.keys() if 
                         itemCategoryDic[k]!=cate]                        
        print i,len(testData)
        testX = getFeatures(testData, itemFeature)    
        predictY = trainModel.predict_proba(testX)
                
        #得到正样本，并标记索引
        positive = []
        ind = 0
        for i in predictY[:,1]:
            if i >0.5:
                positive.append((ind,i))
            ind += 1
        positive = heapq.nlargest(200, positive, key = lambda item : item[1]) 
        positive = [testData[i] for (i,score) in positive[0:200]]
        result.append(positive)
                
    return result

def main():    
    basepath = "../data/offline/"
    itemFile = basepath + 'dim_items.txt'
    itemFeatureFile = basepath + 'item_feature.dat'
    trainFile = basepath + "train_set.txt"
    testPath = basepath + "test_items.txt"
    submitPath = basepath + "fm_submissions.txt"

    itemData = open(itemFile).readlines()
    itemCategoryDic = {}
    for line in itemData:
        tmp = line.split()
        itemCategoryDic[tmp[0]] = tmp[1]

    itemFeature = {}
    itemFeatureData = open(itemFeatureFile).readlines()
    for i in range(0, len(itemFeatureData)):
        line = itemFeatureData[i]
        tmp = line.split()
        item = tmp[0].split('/')[1].split('.')[0]
        feature = [float(tmp[j]) for j in range(1, len(tmp))]
        itemFeature[item] = feature
                
    trainData = []
    trainY = []
     
    #训练集中采样
    sampleRate = 2
    for line in open(trainFile, "r"):
        tmp = line[0:-1].split()
        label = int(tmp[2])
        if label == 1:
            trainData.append((tmp[0], tmp[1]))
        else:
            if random.randint(0,10*sampleRate) % sampleRate != 0:
                continue
            trainData.append((tmp[0], tmp[1]))
        trainY.append(label)
    print len(trainY), len(trainData), config.topicCount
    trainY = np.array(trainY)    
    trainX = getFeatures(trainData, itemFeature)
        
    print "extract train features finished"
          
    rf = RandomForestClassifier(n_estimators=200, n_jobs=config.poolSize)
    trainModel = rf.fit(trainX, trainY)
      
    print "train finish"
      
    testList = generateDic.getTestList(testPath)
    
    args = []    
    results = []
    
    arg = [testList[start:end], itemCategoryDic, trainModel]
    results = [predict(arg)]
    
    resultFile = open(submitPath,"w")
    resultDic = {}
    for result in results:
        for i in result:
            if i[0] not in resultDic:
                resultDic[i[0]] = []
            resultDic[i[0]].append(i[1])
        
    for test in testList:
        resultFile.write(str(i[0]) + " " + ",".join([str(j) for j in resultDic[test]]) + "\n")
                
if __name__ == "__main__":
    main()


