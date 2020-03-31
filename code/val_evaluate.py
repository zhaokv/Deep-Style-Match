import numpy as np

from keras.callbacks import Callback
from sklearn import metrics

import cPickle as pickle


class ValEvaluate(Callback):
    def __init__(self, val_data, result_path,
            batch_size=256, verbose=1):
        self.val_data=val_data
        self.result_file=open(result_path, 'w')
        self.batch_size=batch_size
        self.verbose=verbose

    def on_epoch_end(self, epoch, logs={}):
        self.result_file.write('Result of epoch %s:\n' % (epoch+1))
        y_pred_val = self.model.predict(self.val_data, 
                batch_size=self.batch_size, verbose=self.verbose)
        y_val = self.val_data['pair_out']
        y_pred_val = y_pred_val['pair_out']
        with open('taobao_pred_'+str(epoch+1)+'.dat', 'w') as f:
            pickle.dump((y_val, y_pred_val), f)
        line = 'AUC: %s\n' % (metrics.roc_auc_score(y_val, y_pred_val)*100)
        self.result_file.write(line)
        line = 'Ground Pos: %s, Predict Pos: %s\n' % (int(np.sum(y_val)), int(np.sum(y_pred_val)))
        self.result_file.write(line)
        self.result_file.flush()

