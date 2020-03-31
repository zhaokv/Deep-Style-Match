import time
import numpy as np
import cPickle as pickle

from scipy import sparse
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.ensemble import RandomForestClassifier as RF

import config

from pair_generator import PairGenerator

def main():
    data_dir = '../data/taobao/'
#    data_dir = '../data/amazon/'
    item_path = data_dir+'dim_items.txt'
    sub_item_path = data_dir+'match_item.txt'
    train_pair_path = data_dir+'train_set_1to1.txt'
    if data_dir=='../data/amazon/':
        item_path = data_dir+'dim_items.txt'
        sub_item_path = data_dir+'match_item.txt'
        train_pair_path = data_dir+'train_set_1to1.txt'
    pg = PairGenerator(item_path, sub_item_path, train_pair_path)

    print('Preparing data...', time.ctime())
    (train_data, test_data) = pg.fetch_all(0.2)
    sentence_left = train_data['left_in']
    sentence_right = train_data['right_in']
    pair_label = train_data['pair_out']
    train_num = len(sentence_left)
    I = []
    J = []
    V = []
    train_y = []
    for i in range(0, train_num):
        for j in range(0, len(sentence_left[i])):
            I.append(i)
            J.append(sentence_left[i][j])
            V.append(1)
        for j in range(0, len(sentence_right[i])):
            I.append(i)
            J.append(sentence_right[i][j])
            V.append(1)
        train_y.append(pair_label[i])
    train_x = sparse.coo_matrix((V,(I,J)),shape=(train_num , len(pg.alphabet))).tocsr()

    test_ratio = 0.5
    test_upper = int(len(test_data['left_in'])*test_ratio)
    sentence_left = test_data['left_in']
    sentence_right = test_data['right_in']
    pair_label = test_data['pair_out']
    test_num = len(sentence_left)
    I = []
    J = []
    V = []
    test_y = []
    for i in range(test_upper, test_num):
        for j in range(0, len(sentence_left[i])):
            I.append(i-test_upper)
            J.append(sentence_left[i][j])
            V.append(1)
        for j in range(0, len(sentence_right[i])):
            I.append(i-test_upper)
            J.append(sentence_right[i][j])
            V.append(1)
        test_y.append(pair_label[i])
    test_x = sparse.coo_matrix((V,(I,J)),shape=(test_num-test_upper , len(pg.alphabet))).tocsr()
    print('Start training...', time.ctime())
    #C = NB()
    C = RF(verbose=1, n_jobs=2)
    C.fit(train_x, train_y)
    test_y_pred = C.predict_proba(test_x)
    test_y_pred = [result[1] for result in test_y_pred]
    with open(data_dir.split('/')[-2]+'_rf_pred.dat', 'w') as f:
        pickle.dump((test_y, test_y_pred), f)
    line = 'AUC: %s' % (metrics.roc_auc_score(test_y, test_y_pred)*100)
    print(line)
    line = 'Ground Pos: %s, Predict Pos: %s' % (int(np.sum(test_y)), int(np.sum(test_y_pred)))
    print(line)

if __name__=='__main__':
    main()
