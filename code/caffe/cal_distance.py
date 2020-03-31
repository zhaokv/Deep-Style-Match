import numpy as np
import cPickle as pickle

from sklearn.metrics.pairwise import euclidean_distances as ed

from tqdm import *

featureDic = {}
with open('item_feature.dat') as f:
    featureDic = pickle.load(f)
y = []
pred_y = []
dataDir = '/home/zhaokui/research/KDD/data/taobao/'
testData = open(dataDir+'pro_test_set.txt').readlines()
i = 0
for item in featureDic:
    print(featureDic[item])
    i+=1
    if i==10:
        exit()
for line in tqdm(testData):
    tmp = line.split()
    itemA = tmp[0].split('/')[1].split('.')[0]
    itemB = tmp[1].split('/')[1].split('.')[0]
    #print(featureDic[itemA], featureDic[itemB])
    print(ed(featureDic[itemA], featureDic[itemB]))
    i += 1
    if i==10:
        break
