import time
import numpy as np
import theano

from sklearn import metrics

from keras.models import Graph
from keras.backend import theano_backend as K
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping

from pair_generator import PairGenerator
from val_evaluate import ValEvaluate
import config

if config.debug:
    theano.config.optimizer='None'
    theano.config.exception_verbosity='high'

def main():
    data_dir = '../data/taobao/'
#    data_dir = '../data/amazon/'
    item_path = data_dir+'dim_items.txt'
    sub_item_path = data_dir+'match_item.txt'
    train_pair_path = data_dir+'train_set_1to1.txt'
    if data_dir=='../data/amazon/':
        item_path = data_dir+'dim_items_all.txt'
        sub_item_path = data_dir+'match_item_all.txt'
        train_pair_path = data_dir+'train_set_1to1_all.txt'
    pg = PairGenerator(item_path, sub_item_path, train_pair_path)

    graph = Graph()
    graph.add_input(name='left_in', input_shape=(pg.max_len,), dtype='int')
    graph.add_input(name='right_in', input_shape=(pg.max_len,), dtype='int')

    activation='relu'
    filter1_length=3
    pool1_length=4
    filter2_length=2
    pool2_length=2
    nb_filter=100
    nb_epoch=5
    optimizer='adagrad'

    graph.add_shared_node(
        Embedding(input_dim=len(pg.alphabet), 
            output_dim=config.w2vSize, 
            input_length= pg.max_len, 
            mask_zero=True,
        weights=[pg.w2v_weight]),
        name='embedding',
        inputs=['left_in', 'right_in'])
    graph.add_shared_node(
        ZeroPadding1D(
            padding=filter1_length-1
        ),
        name='padding1',
        inputs=['embedding'])
    graph.add_shared_node(
        Convolution1D(nb_filter=nb_filter, 
            filter_length=filter1_length,
            border_mode='valid', activation=activation, 
            subsample_length=1
        ), 
        name='conv1', 
        inputs=['padding1'])
    graph.add_shared_node(
        MaxPooling1D(pool_length=pool1_length
        ),
        name='max1', 
        inputs=['conv1'])
    graph.add_shared_node(
        ZeroPadding1D(
            padding=filter2_length-1
        ),
        name='padding2',
        inputs=['max1'])
    graph.add_shared_node(
        Convolution1D(nb_filter=nb_filter, 
            filter_length=filter2_length, 
            border_mode='valid', activation=activation, 
            subsample_length=1
        ), 
        name='conv2',
        inputs=['padding2'])
    graph.add_shared_node(
        MaxPooling1D(pool_length=pool2_length
        ),
        name='max2', merge_mode=None,
        inputs=['conv2'])
    graph.add_shared_node(
        Dropout(0.2),
        name='dropout', merge_mode=None,
        inputs=['max2'])
    graph.add_shared_node(
        Flatten(), 
        name='flatten', merge_mode=None,
        inputs=['dropout'])
    graph.add_shared_node(
        Dense(output_dim=config.feature_dim,
            activation=activation
        ), 
        name='dense1', merge_mode=None,
        inputs=['flatten'], 
        outputs=['dense1_left', 'dense1_right'])
    
    #graph.add_shared_node(
    #    Dense(output_dim=len(pg.cat2idx),
    #        activation='softmax'),
    #    name='dense2', 
    #    inputs=['dense1'], 
    #    outputs=['dense2_output1', 'dense2_output2'])
    graph.add_node(
        Dense(output_dim=config.feature_dim,
            b_constraint=maxnorm(m=0),
            activation='linear'
        ), 
        name='dense3',  
        input='dense1_left')
    graph.add_node(
        Dense(output_dim=1,
            activation='sigmoid'
        ), 
        name='dense4',  
        inputs=['dense3', 'dense1_right'],
        merge_mode='dot')
    #graph.add_output(name='left_out', input='dense2_output1')
    #graph.add_output(name='right_out', input='dense2_output2')
    graph.add_output(name='pair_out', input='dense4')
    graph.compile(optimizer=optimizer, 
        loss={#'left_out': 'categorical_crossentropy',
            #'right_out': 'categorical_crossentropy',
            'pair_out': 'binary_crossentropy'},
        #loss_weight={'left_out': 0.0,
        #    'right_out': 0.0,
        #    'pair_out': 1}
        )
    print(graph.summary())
    print('Preparing data...', time.ctime())
    (train_data, test_data) = pg.fetch_all(0.1)
    print('Start training...', time.ctime())
    callbacks=[ValEvaluate(test_data, 'result.txt'), 
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
    graph.fit(data=train_data,
        validation_split=0.1111111111, 
        callbacks=callbacks, 
        batch_size=256, nb_epoch=nb_epoch,verbose=1)

if __name__=='__main__':
    main()
