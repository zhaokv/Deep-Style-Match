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
    #data_dir = '../data/amazon/'
    item_path = data_dir+'dim_items.txt'
    sub_item_path = data_dir+'match_item.txt'
    train_pair_path = data_dir+'train_set_1to1.txt'
    pg = PairGenerator(item_path, sub_item_path, train_pair_path)

    print('Preparing data...', time.ctime())
    label = '1'
    if data_dir=='../data/amazon/':
        label = '0'
    (train_data, test_data) = pg.fetch_all_topic(label, 0.2)
    train_x = np.array(train_data['pair_in'])
    train_y = np.array(train_data['pair_out'])
    test_num = len(test_data['pair_in'])
    test_upper = int(test_num*0.5)
    test_x = np.array(test_data['pair_in'][test_upper:])
    test_y = np.array(test_data['pair_out'][test_upper:])
    print('Start training...', time.ctime())
    #C = NB()
    C = RF(verbose=1, n_jobs=2)
    C.fit(train_x, train_y)
    test_y_pred = C.predict_proba(test_x)
    test_y_pred = [result[1] for result in test_y_pred]
    with open(data_dir.split('/')[-2]+'_lda_pred.dat', 'w') as f:
        pickle.dump((test_y, test_y_pred), f)
    line = 'AUC: %s' % (metrics.roc_auc_score(test_y, test_y_pred)*100)
    print(line)
    line = 'Ground Pos: %s, Predict Pos: %s' % (int(np.sum(test_y)), int(np.sum(test_y_pred)))
    print(line)
    

if __name__=='__main__':
    main()
