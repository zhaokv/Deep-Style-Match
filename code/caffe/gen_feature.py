import sys
caffeRoot = '/home/zhaokui/code/caffe/'
sys.path.insert(0, caffeRoot+'python')
import caffe

import numpy as np
import cPickle as pickle
import copy
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import roc_auc_score as auc

from tqdm import *

modelDir = '/home/zhaokui/research/KDD/code/caffe/siamese_googlenet/'
caffe.set_mode_gpu()
net = caffe.Classifier(modelDir+'deploy1.prototxt',
                       modelDir+'googlenet_iter_60000.caffemodel',
                       (256, 256),
                       np.load('item_mean.npy').mean(1).mean(1),
                       None,
                       224,
                       (2,1,0))


itemDir = '/home/zhaokui/research/KDD/data/taobao/'
itemData = open(itemDir+'pro_test_item.txt').readlines()
featureDic = {}
for item in tqdm(itemData):
    name = item.split('/')[1].split('.')[0]
    image = caffe.io.load_image(itemDir+item.split()[0])
    net.predict([image], False)
    featureDic[name] = copy.deepcopy(net.blobs['loss3f'].data[0])

y = []
pred_y = []
dataDir = '/home/zhaokui/research/KDD/data/taobao/'
testData = open(dataDir+'pro_test_set.txt').readlines()
for line in tqdm(testData):
    tmp = line.split()
    itemA = tmp[0].split('/')[1].split('.')[0]
    itemB = tmp[1].split('/')[1].split('.')[0]
    y.append(int(tmp[2]))
    pred_y.append(ed(np.array(featureDic[itemA]).reshape(1,-1), 
        np.array(featureDic[itemB]).reshape(1, -1))[0][0])

with open('predict.dat', 'w') as f:
    pickle.dump((y, pred_y), f)

with open('item_feature.dat', 'w') as f:
    pickle.dump(featureDic, f)

print(auc(y, pred_y))

