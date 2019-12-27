from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import pickle
import pandas as pd
import math
import tensorflow as tf
from hanziconv import HanziConv
import json
import gensim
import datetime
from gensim.models import KeyedVectors
# FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("delimiter","\001","delimiter of columns")
# tf.app.flags.DEFINE_integer("line_per_file",500000,"lines per result file")
# tf.app.flags.DEFINE_string("emb_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/SummaRuNNer/checkpoint/model2","word2vec's vocabulary and vectors")


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        # model = KeyedVectors.load("./checkpoint/model2", mmap='r')
        if token not in self._token2index:
            # allocate new index for this token
            # if token == 'unk' or token == '{' or token == '}' or token == ' ':
            #     index = len(self._token2index)
            #     self._token2index[token] = index
            #     self._index2token.append(token)
            # if token not in model:
            #     index = 0
            #     self._token2index[token] = index
            #     self._index2token.append(token)
            index = len(self._token2index)
            # print(type(index))
            # self._token2index[token] = long(index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    def feed_word(self, token):
        if token not in self._token2index:
            index = 0
            self._token2index[token] = index
            self._index2token.append(token)
        return self._token2index[token]
        
    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        # print(token)
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None): 
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data_excel(data_dir, max_doc_length=15, max_sent_length=50):
    word_vocab = Vocab()
    word_vocab.feed('unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    print("start loading embeddings : " + str(datetime.datetime.now()))
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    print("loading done : " + str(datetime.datetime.now()))
    # count = 0
    for word in model.vocab:
        # count += 1
        # if count % 10000 == 0:
        #     print(count)
        word_vocab.feed(word)
    df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    # df = pd.read_excel('./data/split_90/6.xlsx', sheet_name='Sheet0')
    # headlineids = df['headlineid'].tolist()
    # contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    # texts = df['feed_text'].tolist()
    idset = {}
    contentset = {}
    print("start processing data : " + str(datetime.datetime.now()))
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        sentences = content.replace('@', '').replace('䧪', '').replace(' ', '')
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if text in sentences:
            if headlineid in idset:
                if len(str(idset[headlineid])) > len(str(text)):
                    continue
                else:
                    idset[headlineid] = str(text)
                    contentset[headlineid] = content
            else:
                idset[headlineid] = str(text)
                contentset[headlineid] = content
    # print(len(idset))
    actual_max_doc_length = 0
    print("start labeling : " + str(datetime.datetime.now()))
    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    ids = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for headlineid in idset:
        if index % 9 == 0 and index % 2 == 0:
            fname = "test"
        elif index % 7 == 0 and index % 2 == 0:
            fname = "valid"
        else:
            fname = "train"
        word_doc = []
        label_doc = []
        text = idset[headlineid]
        content = contentset[headlineid]
        sentences = content.split('䧪')
        index += 1
        flag = 0
        # if index % 50 == 0:
        #     print(index)
        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence))
            if text in sentence.replace(' ', '').replace('@', '') or sentence.replace(' ', '').replace('@', '') in text or sentence[:-1].replace(' ', '').replace('@', '') in text:
                label = 1
                # pos_label += 1
                flag = 1
            else:
                label = 0
                # neg_label += 1
            label_doc.append(label)
            sent_list = sentence.split("@")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed_word(c) for c in ['{'] + sent_list + ['}']]
            # print(word_array)
            word_doc.append(word_array)
        if flag == 0:
            continue
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        # print(word_tokens)
        labels[fname].append(label_doc)
        ids[fname].append(headlineid)

    actual_max_doc_length = max_doc_length
    print('total index', index)
    print('positive label', pos_label)
    print('negative label', neg_label)
    assert actual_max_doc_length <= max_doc_length

    print("max_doc_len:", actual_max_doc_length)
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    id_tensors = {}
    print("start making tensors : " + str(datetime.datetime.now()))
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)
        id_tensors[fname] = np.zeros([len(ids[fname])], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                # print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

        for i, id_doc in enumerate(ids[fname]):
            id_tensors[fname][i] = id_doc
    return word_vocab, word_tensors, actual_max_doc_length, label_tensors, id_tensors


# This function is used to get the word embedding of current words from
def get_embed(word_vocab):
    # model = gensim.models.Word2Vec.load("./checkpoint/model1")
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    f = codecs.open("./oov_list_mm", "w", "utf-8")
    # print("space: ", model[" "])
    embed_list = []
    # uni_list = [0 for i in range(150)]
    uni_list = [0 for i in range(100)]
    embed_list.append(uni_list)
    # print (embed_list)
    vocab_dict = word_vocab._token2index
    vocab_sorted = sorted(vocab_dict.items(), key=lambda asd: asd[1], reverse=False)
    i = 0
    n_sum = 0
    w_dict = {}
    unk_embed = np.random.uniform(-0.25, 0.25, 100).round(6).tolist()
    for item, index in vocab_sorted:
        # print("space:", item,index, ".")
        n_sum += 1
        if index == 0:
            # print(0, item)
            embed_list.append(unk_embed)
        elif item not in model:
            f.write(item)
            f.write("\n")
            print('oov: ', item, index)
            # i += 1
            # if item not in w_dict.keys():
            #     w_dict[item] = 1
            # else:
            #     w_dict[item] += 1
            # embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
            embed_list.append(np.random.uniform(-0.25, 0.25, 100).round(6).tolist())
            # embed_list.append(uni_list)
        else:
            # print(list(model[item]))
            embed_list.append(list(model[item]))
    print("no in:", i)
    print("all:", n_sum)
    f.close()
    # print(embed_list[0])
    embedding_array = np.array(embed_list, np.float32)
    print("no in:", i)
    print("all:", n_sum)
    # f = codecs.open("./fre_sum", "w", "utf-8")
    # for key, value in w_dict.items():
    #     f.write(key)
    #     f.write("\t")
    #     f.write(str(value))
    #     f.write("\n")
    # f.close()
    return embedding_array

class DataReader:

    def __init__(self, word_tensor, label_tensor, id_tensor, batch_size):

        length = word_tensor.shape[0]
        #print (length)
        doc_length = word_tensor.shape[1]
        #print (doc_length)
        sent_length = word_tensor.shape[2]
        #print (sent_length)

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        #print (clipped_length)
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]
        id_tensor = id_tensor[:clipped_length]
        print(word_tensor.shape)

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        #print(x_batches.shape)
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])
        z_batches = id_tensor.reshape([batch_size, -1])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        #print(x_batches.shape)
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        z_batches = np.transpose(z_batches, axes=(1, 0))

        self._x_batches = list(x_batches)
        #print(len(self._x_batches))
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        #print (self.length)
        self.batch_size = batch_size
        self.max_sent_length = sent_length
        self._z_batches = list(z_batches)
        assert len(self._x_batches) == len(self._z_batches)
    def iter(self):

        for x, y, z in zip(self._x_batches, self._y_batches, self._z_batches):
            yield x, y, z

    def __len__(self):

        return len(self._x_batches)

def load_test(data_dir, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed('unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    for word in model.vocab:
        word_vocab.feed(word)
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    df = pd.read_excel('./data/split_90/14.xlsx', sheet_name='Sheet1')
    headlineids = df['headlineid'].tolist()
    contents = df['sentencesegs'].tolist()
    contents_raw = df['content'].tolist()
    texts = df['feed_text'].tolist()
    idset = {}
    contentset = {}
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        if headlineid == 246479829:
            print('or', text)
        sentences = content.replace('@', '').replace('䧪', '').replace(' ', '')
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if text in sentences:
            if headlineid in idset:
                if len(str(idset[headlineid])) > len(str(text)):
                    # print(1, i, text)
                    continue
                else:
                    idset[headlineid] = str(text)
                    contentset[headlineid] = content
            else:
                idset[headlineid] = str(text)
                contentset[headlineid] = content
    print(len(idset))
    actual_max_doc_length = 0
    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    ids = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for headlineid in idset:
        fname = "test"
        word_doc = []
        label_doc = []
        content = contentset[headlineid]
        sentences = content.split('䧪')
        index += 1
        for sentence in sentences:
            label = sentence.replace('@','')
            label_doc.append(label)
            sent_list = sentence.split('@')
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
            word_doc.append(word_array)
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        labels[fname].append(label_doc)
        ids[fname].append(headlineid)

    actual_max_doc_length = max_doc_length
    print('total index', index)
    assert actual_max_doc_length <= max_doc_length

    print("max_doc_len:", actual_max_doc_length)
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    id_tensors = {}

    for fname in ['test']:
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        # label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)
        # arr = [len(labels[fname])][actual_max_doc_length]
        # for i in range(len(labels[fname])):
        #     for j in range(actual_max_doc_length):
        #         label_tensors[fname][i][j] = ''
        # label_tensors[fname] = [[j for j in range(max_doc_length)] for i in range(len(labels[fname])]
        label_tensors[fname] = []
        id_tensors[fname] = np.zeros([len(ids[fname])], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                # print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            tmp = []
            for j in range(max_doc_length):
                if j < len(label_doc):
                    tmp.append(label_doc[j])
                else:
                    tmp.append('')
            label_tensors[fname].append(tmp)
        label_tensors[fname] = np.asarray(label_tensors[fname])
            # label_tensors[fname].append(label_doc)
            # tmp1 = []
            # for j, tmp in enumerate(label_doc):
            #     tmp1.append(tmp)
            # label_tensors[fname].append(tmp1)
            # for j, label in enumerate(label_doc[i]):
            # label_tensors[fname][i][0:len(label_doc)] = label_doc
                # print(labels[fname][i])
                # print(labels[fname][i][j])
                # label_tensors[fname][i].append(label_doc[j])

        for i, id_doc in enumerate(ids[fname]):
            id_tensors[fname][i] = id_doc
    return word_vocab, word_tensors, actual_max_doc_length, label_tensors, id_tensors