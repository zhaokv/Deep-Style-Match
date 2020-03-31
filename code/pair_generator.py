import gensim
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils

import random
import time
import cPickle as pickle

from alphabet import Alphabet
import config

def word2vec(sentence_list):
    model = gensim.models.Word2Vec(sentence_list, 
            size=config.w2vSize, 
            min_count = 0, workers=4)
    return model

def lda(sentence_list):
    texts = sentence_list
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = gensim.models.ldamodel.LdaModel(corpus, 
            num_topics=config.topicNum)
    return (model, dictionary)

def add_to_vocab(sentence_list, alphabet):
    for sentence in sentence_list:
        for word in sentence:
            alphabet.add(word)
    return alphabet

class PairGenerator(object):
    '''Generate minibatches with
    realtime data combination
    '''
    def __init__(self, item_path, sub_item_path, pair_path, 
            split_c = ','):
        self.__dict__.update(locals())
        
        print('Loading title and category information...', time.ctime())
        sub_item_set = set()
        for line in open(sub_item_path).readlines():
            sub_item_set.add(line.split()[0])
        self.item_title = {}
        self.item_cat = {}
        self.cat2idx = {}
        self.max_len = 0
        sentence_list = []
        for line in open(item_path).readlines():
            tmp = line.split()
            item = tmp[0]
            cat = tmp[1]
            if cat not in self.cat2idx:
                self.cat2idx[cat] = len(self.cat2idx)
            title = tmp[2].split(split_c)
            self.item_title[item] = title
            self.item_cat[item] = self.cat2idx[cat]
            if item in sub_item_set:
                sentence_list.append(title)
                self.max_len = min(config.max_len, max(self.max_len, len(title)))
        print(('%s items' % len(sentence_list)), time.ctime())
        
        print('Generating alphabet...', time.ctime())
        self.alphabet = Alphabet()
        add_to_vocab(sentence_list, self.alphabet)
        print(('%s words' % len(self.alphabet)), time.ctime())

        print('Generating weight from word2vec model...', time.ctime())
        self.sentence_list = sentence_list
        w2v_model = word2vec(sentence_list)
        self.w2v_weight = np.zeros((len(self.alphabet), config.w2vSize))
        for word, idx in self.alphabet.iteritems():
            if word in w2v_model.vocab:
                self.w2v_weight[idx] = w2v_model[word]
        
        print('Loading pairs ...', time.ctime())
        self.pair_list = open(pair_path).readlines()

    def batch(self, pair_list):
        left_in = []
        right_in = []
        left_out = []
        right_out = []
        pair_out = []
        for line in pair_list:
            tmp = line.split(',')
            left = tmp[0]
            right = tmp[1]
            pair = int(tmp[2])
            left_in.append([self.alphabet[word] for word in self.item_title[left]])
            right_in.append([self.alphabet[word] for word in self.item_title[right]])
            #left_out.append(self.item_cat[left])
            #right_out.append(self.item_cat[right])
            pair_out.append(pair)

        return {'left_in': sequence.pad_sequences(left_in, maxlen=self.max_len), 
                'right_in': sequence.pad_sequences(right_in, maxlen=self.max_len), 
                #'left_out': np_utils.to_categorical(left_out, nb_classes=len(self.cat2idx)), 
                #'right_out': np_utils.to_categorical(right_out, nb_classes=len(self.cat2idx)), 
                'pair_out': np.array(pair_out)}

    def fetch_all(self, val_split=0.2, pair_list=None):
        if not pair_list:
            pair_list = self.pair_list
        upper_bound = int(len(pair_list)*(1-val_split))
        train_data = self.batch(pair_list[0:upper_bound])
        val_data =self.batch(pair_list[upper_bound:])
        return (train_data, val_data)
        
    def batch_topic(self, item_vector_dic, pair_list):
        pair_in = []
        pair_out = []
        for line in pair_list:
            tmp = line.split(',')
            left = tmp[0]
            right = tmp[1]
            pair = int(tmp[2])
            vector_left = [0.0]*config.w2vSize
            vector_right = [0.0]*config.w2vSize
            for (pos, value) in item_vector_dic[left]:
                vector_left[pos] = value
            for (pos, value) in item_vector_dic[right]:
                vector_right[pos] = value
            pair_in.append(vector_left+vector_right)
            pair_out.append(pair)

        return {'pair_in': pair_in, 
                'pair_out': pair_out}
    
    def fetch_all_topic(self, mark, val_split=0.2, pair_list=None):
        item_vector_dic = {}
        if config.fresh:
            print('Generating topic model...', time.ctime())
            (model, dictionary) = lda(self.sentence_list)

            print('Generating topic vector...', time.ctime())
            for item in self.item_title:
                title = self.item_title[item]
                doc_bow = dictionary.doc2bow(title)
                item_vector_dic[item] = model[doc_bow]
            with open(mark+'topic.tmp', 'w') as f:
                pickle.dump(item_vector_dic, f)
        else:
            print('Loading topic vector...', time.ctime())
            with open(mark+'topic.tmp') as f:
                item_vector_dic = pickle.load(f)

        if not pair_list:
            pair_list = self.pair_list
        upper_bound = int(len(pair_list)*(1-val_split))
        train_data = self.batch_topic(item_vector_dic, pair_list[0:upper_bound])
        val_data =self.batch_topic(item_vector_dic, pair_list[upper_bound:])
        return (train_data, val_data)
        
    def flow(self, batch_size=32, shuffle=False, seed=None):
        if seed:
            random.seed(seed)

        if shuffle:
            random.shuffle(self.pair_list)

        b = 0
        pair_num = len(self.pair_list)
        while 1:
            current_index = (b * batch_size) % pair_num
            if pair_num >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = pair_num - current_index
            yield self.batch(self.pair_list[current_index:current_index+current_batch_size])
