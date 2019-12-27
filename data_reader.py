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
from gensim.models import KeyedVectors

#--------------------------------------------------------------------
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("delimiter","\001","delimiter of columns")
tf.app.flags.DEFINE_integer("line_per_file",500000,"lines per result file")
tf.app.flags.DEFINE_string("emb_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/SummaRuNNer/checkpoint/model2","word2vec's vocabulary and vectors")

# This class is used to create the vocab
class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            #print(type(index))
            #self._token2index[token] = long(index)
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
        #print(token)
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

# This function is used to load the data from
def load_data(data_dir, max_doc_length=15, max_sent_length=50):

    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    for fname in ('train', 'valid', 'test'):
        #print('reading', fname)
        pname = os.path.join(data_dir, fname)
        for dname in os.listdir(pname):
            if fname == "test":
                h.write(dname)
                h.write("\n")
                #print (dname)
            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                word_doc = []
                label_doc = []
                for line in f.readlines():
                    #print (line)
                    line = line.strip()
                    #line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')
                    if "\t" not in line:
                        continue
                    #l_dict = json.loads(line)
                    label, sent = line.split('\t')
                    if label != "0" and label != "1":
                        label = "1"
                    #print(sent)
                    label_doc.append(label)
                    #print (sent)
                    #sent_list = sent.split(",")
                    sent_list = sent.split(" ")
                    #sent = json.loads(sent).keys()
                    #print(sent)
                    #sent--word list
                    #print(len(sent))
                    if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent_list = sent_list[:max_sent_length-2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
                    #print(word_array)
                    word_doc.append(word_array)

                if len(word_doc) > max_doc_length:
                    word_doc = word_doc[:max_doc_length]
                    label_doc = label_doc[:max_doc_length]
                #print(actual_max_doc_length)
                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)
                #print(word_tokens)
                labels[fname].append(label_doc)
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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length], dtype=np.int64)
        #print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)
 
        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors


# load data function for excel file format
def load_data_excel_v1(data_dir, max_doc_length=15, max_sent_length=50):
    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    df = pd.read_excel('./data/split_90/4.xlsx', sheet_name='Sheet0')
    headlineids = df['headlineid'].tolist()
    contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    texts = df['feed_text'].tolist()
    idset = {}
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        sentences = content.replace('@', '').replace('䧪', '').replace(' ', '')
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if text in sentences:
            if headlineid in idset:
                if len(str(idset[headlineid])) > len(str(text)):
                    continue
                else:
                    idset[headlineid] = str(text)
            else:
                idset[headlineid] = str(text)
    del_headlineids = []
    del_texts = []
    del_contents = []
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        # if headlineid == 155241197:
        #     print(1)
        #     print(text)
        if headlineid == 241842344:
            print(0)
        if (not isinstance(text, str)) and (math.isnan(text)):
            if headlineid == 241842344:
                print(1)
            # if headlineid == 155241197:
            #     print(content)
            #     print(text)
            del headlineids[i]
            del texts[i]
            del contents[i]
            # del contents_raw[i]
        elif headlineid not in idset:
            del headlineids[i]
            del texts[i]
            del contents[i]
            # del contents_raw[i]
        elif str(text) != idset[headlineid]:
            # if headlineid == 241842344:
            #     print(3)
            # if headlineid == 155241197:
            #     print(3)
            #     print(text)
            del headlineids[i]
            del texts[i]
            del contents[i]
            # del contents_raw[i]
    # for ele in del_headlineids:
    #     headlineids.remove(ele)
    # for ele in del_texts:
    #     texts.remove(ele)
    # for ele in del_contents:
    #     contents.remove(ele)

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for (headlineid, content, text) in zip(headlineids, contents, texts):
        if index % 9 == 0 and index != 0:
            fname = "test"
            #h.write(str(headlineid))
            #h.write("\n")
        elif index % 7 == 0 and index != 0:
            fname = "valid"
        else:
            fname = "train"
        word_doc = []
        label_doc = []
        sentences = content.replace('@', ' ').split('䧪')
        # if headlineid == 155241197:
        #     print(content)
        #     print(len(sentences))
        #     print(text)
        #     print((not isinstance(text, str)) and (math.isnan(text)))
        index += 1
        flag = 0
        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence))
            # print(sentence)
            # print(text in sentence.replace(' ', ''))
            if text in sentence.replace(' ', '') or sentence.replace(' ', '') in text or sentence[:-1].replace(' ', '') in text:
                label = 1
                pos_label += 1
                flag = 1
            else:
                label = 0
                neg_label += 1
            # if headlineid == 155241197:
            #     print(label)
            #     print(sentence)
            label_doc.append(label)
            sent_list = sentence.split(" ")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                #print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def load_data_excel_v2(data_dir, max_doc_length=15, max_sent_length=50):
    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    df = pd.read_excel('./data/split_90/6.xlsx', sheet_name='Sheet0')
    headlineids = df['headlineid'].tolist()
    contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    texts = df['feed_text'].tolist()
    idset = {}
    contentset = {}
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
    del_headlineids = []
    del_texts = []
    del_contents = []
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        # if headlineid == 155241197:
        #     print(1)
        #     print(text)
        if (not isinstance(text, str)) and (math.isnan(text)):
            del_headlineids.append(headlineid)
            del_texts.append(text)
            del_contents.append(content)
        elif headlineid not in idset:
            del_headlineids.append(headlineid)
            del_texts.append(text)
            del_contents.append(content)
        elif str(text) != idset[headlineid]:
            del_headlineids.append(headlineid)
            del_texts.append(text)
            del_contents.append(content)
    for ele in del_headlineids:
        headlineids.remove(ele)
    for ele in del_texts:
        texts.remove(ele)
    for ele in del_contents:
        contents.remove(ele)

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for headlineid in idset:
        if index % 9 == 0 and index != 0:
            fname = "test"
            #h.write(str(headlineid))
            #h.write("\n")
        elif index % 7 == 0 and index != 0:
            fname = "valid"
        else:
            fname = "train"
        word_doc = []
        label_doc = []
        text = idset[headlineid]
        content = contentset[headlineid]
        sentences = content.replace('@', ' ').split('䧪')
        # if headlineid == 155241197:
        #     print(content)
        #     print(len(sentences))
        #     print(text)
        #     print((not isinstance(text, str)) and (math.isnan(text)))
        index += 1
        flag = 0
        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence))
            if text in sentence.replace(' ', '') or sentence.replace(' ', '') in text or sentence[:-1].replace(' ', '') in text:
                label = 1
                pos_label += 1
                flag = 1
            else:
                label = 0
                neg_label += 1
            # if headlineid == 155241197:
            #     print(label)
            #     print(sentence)
            label_doc.append(label)
            sent_list = sentence.split(" ")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
            # print(word_array)
            word_doc.append(word_array)
        if flag == 0:
            # print(sentences)
            # print(text)
            continue
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        # print(word_tokens)
        labels[fname].append(label_doc)

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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                #print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def load_data_excel_v3(data_dir, max_doc_length=15, max_sent_length=50):
    word_vocab = Vocab()
    word_vocab.feed('unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    df = pd.read_excel('./data/split_90/6.xlsx', sheet_name='Sheet0')
    headlineids = df['headlineid'].tolist()
    contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    texts = df['feed_text'].tolist()
    idset = {}
    contentset = {}
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

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for headlineid in idset:
        if index % 9 == 0 and index % 2 == 0:
            fname = "test"
            #h.write(str(headlineid))
            #h.write("\n")
        elif index % 7 == 0 and index % 2 == 0:
            fname = "valid"
        else:
            fname = "train"
        word_doc = []
        label_doc = []
        text = idset[headlineid]
        content = contentset[headlineid]
        sentences = content.replace('@', ' ').split('䧪')
        # if headlineid == 155241197:
        #     print(content)
        #     print(len(sentences))
        #     print(text)
        #     print((not isinstance(text, str)) and (math.isnan(text)))
        index += 1
        flag = 0
        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence))
            if text in sentence.replace(' ', '') or sentence.replace(' ', '') in text or sentence[:-1].replace(' ', '') in text:
                label = 1
                pos_label += 1
                flag = 1
            else:
                label = 0
                neg_label += 1
            label_doc.append(label)
            sent_list = sentence.split(" ")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
            # print(word_array)
            word_doc.append(word_array)
        if flag == 0:
            # print(sentences)
            # print(text)
            continue
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        # print(word_tokens)
        labels[fname].append(label_doc)

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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                #print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def load_test_v1(data_dir, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')
    df = pd.read_excel('./data/split_90/3.xlsx', sheet_name='Sheet0')
    headlineids = df['headlineid'].tolist()
    contents = df['sentencesegs'].tolist()
    contents_raw = df['content'].tolist()
    texts = df['feed_text'].tolist()
    idset = {}
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        sentences = content.replace('@', ' ').replace('䧪', ' ').replace(' ', '')
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if text in sentences:
            if headlineid in idset:
                if len(str(idset[headlineid])) > len(str(text)):
                    continue
                else:
                    idset[headlineid] = str(text)
            else:
                idset[headlineid] = str(text)
    print('before', len(texts))
    for i, (headlineid, text) in enumerate(zip(headlineids, texts)):
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if (not isinstance(text, str)) and (math.isnan(text)):
            del headlineids[i]
            del texts[i]
            del contents[i]
            del contents_raw[i]
            print(1, headlineid)
        elif headlineid not in idset:
            del headlineids[i]
            del texts[i]
            del contents[i]
            del contents_raw[i]
            print(2, headlineid)
            continue
        elif str(text) != idset[headlineid]:
            del headlineids[i]
            del texts[i]
            del contents[i]
            del contents_raw[i]
            print(3, headlineid)
    print('after', len(texts))
    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for (headlineid, content, text) in zip(headlineids, contents, texts):
        fname = "test"
        word_doc = []
        label_doc = []
        sentences = content.replace('@', ' ').split('䧪')
        index += 1

        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence)).replace(' ', '')
            if text in sentence or sentence in text or sentence[:-1] in text:
                label = 1
                pos_label += 1
            else:
                label = 0
                neg_label += 1
            label_doc.append(label)
            sent_list = sentence.split(" ")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
            # print(word_array)
            word_doc.append(word_array)

        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        # print(word_tokens)
        labels[fname].append(label_doc)

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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                #print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def load_test_v2(data_dir, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')
    df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    # df = pd.read_excel('./data/split_90/14.xlsx', sheet_name='Sheet1')
    # headlineids = df['headlineid'].tolist()
    # contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    # texts = df['feed_text'].tolist()
    idset = {}
    contentset = {}
    for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
        if headlineid == 246479829:
            print('or', text)
        sentences = content.replace('@', '').replace('䧪', '').replace(' ', '')
        text = HanziConv.toSimplified(str(text)).replace(' ', '')
        if headlineid == 246479829:
            print('af', text)
        if text in sentences:
            if headlineid in idset:
                if len(str(idset[headlineid])) > len(str(text)):
                    print(1, i, text)
                    continue
                else:
                    idset[headlineid] = str(text)
                    contentset[headlineid] = content
            else:
                idset[headlineid] = str(text)
                contentset[headlineid] = content
        else:
            print(2, i, text)
    print(len(idset))
    # del_headlineids = []
    # del_texts = []
    # del_contents = []
    # for i, (headlineid, text, content) in enumerate(zip(headlineids, texts, contents)):
    #     # if headlineid == 155241197:
    #     #     print(1)
    #     #     print(text)
    #     if (not isinstance(text, str)) and (math.isnan(text)):
    #         print(1, i)
    #         del_headlineids.append(headlineid)
    #         del_texts.append(text)
    #         del_contents.append(content)
    #     elif headlineid not in idset:
    #         print(2, text)
    #         del_headlineids.append(headlineid)
    #         del_texts.append(text)
    #         del_contents.append(content)
    #     elif str(text) != idset[headlineid]:
    #         print(3, text)
    #         del_headlineids.append(headlineid)
    #         del_texts.append(text)
    #         del_contents.append(content)
    # for ele in del_headlineids:
    #     headlineids.remove(ele)
    # for ele in del_texts:
    #     texts.remove(ele)
    # for ele in del_contents:
    #     contents.remove(ele)
    # print(del_headlineids)
    # print(del_texts)
    # print(len(contents))
    # print(len(texts))
    # print(len(idset))
    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    index = 0
    pos_label = 0
    neg_label = 0
    for headlineid in idset:
        fname = "test"
        word_doc = []
        label_doc = []
        text = idset[headlineid]
        content = contentset[headlineid]
        sentences = content.replace('@', ' ').split('䧪')
        # if headlineid == 155241197:
        #     print(content)
        #     print(len(sentences))
        #     print(text)
        #     print((not isinstance(text, str)) and (math.isnan(text)))
        index += 1
        flag = 0
        for sentence in sentences:
            text = HanziConv.toSimplified(str(text)).replace(' ', '')
            sentence = HanziConv.toSimplified(str(sentence))
            if text in sentence.replace(' ', '') or sentence.replace(' ', '') in text or sentence[:-1].replace(' ', '') in text:
                label = 1
                pos_label += 1
                flag = 1
            else:
                label = 0
                neg_label += 1
            # if headlineid == 155241197:
            #     print(label)
            #     print(sentence)
            label_doc.append(label)
            sent_list = sentence.split(" ")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
            # print(word_array)
            word_doc.append(word_array)
        if flag == 0:
            # print(sentences)
            # print(text)
            continue
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        # print(word_tokens)
        labels[fname].append(label_doc)

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
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                #print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def load_test_v3(data_dir, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed('<unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    for word in model.vocab:
        word_vocab.feed(word)
    df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    # df = pd.read_excel('./data/split_90/14.xlsx', sheet_name='Sheet1')
    # headlineids = df['headlineid'].tolist()
    # contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    # texts = df['feed_text'].tolist()
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
                    print(1, i, text)
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
        content = idset[headlineid]
        sentences = content.split('䧪')
        index += 1
        for sentence in sentences:
            label = 0
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

def load_from_fileList(fileList, write_index, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed('unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    # df = pd.read_excel('./data/split_90/14.xlsx', sheet_name='Sheet1')
    # headlineids = df['headlineid'].tolist()
    # contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    # texts = df['feed_text'].tolist()
    idset = {}
    lines_list = []
    index_of_line = 0
    shouldEnd = False
    for train_file in fileList:
        lines = tf.gfile.GFile(train_file).readlines()
        for i, line in enumerate(lines):
            index_of_line = index_of_line + 1
            if index_of_line > write_index and index_of_line <= write_index + FLAGS.line_per_file:
                line = line.replace("\n", "")
                lines_list.append(line)
                lines_splits = line.split(FLAGS.delimiter)
                headlineid = lines_splits[0]
                content = lines_splits[3]
                idset[headlineid] = content
    if index_of_line < write_index+FLAGS.line_per_file:
        shouldEnd = True
    actual_max_doc_length = 0
    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    ids = collections.defaultdict(list)
    index = 0
    for headlineid in idset:
        fname = "test"
        word_doc = []
        label_doc = []
        content = idset[headlineid]
        sentences = content.replace('@', ' ').split('䧪')
        index += 1
        for sentence in sentences:
            label = 0
            label_doc.append(label)
            sent_list = sentence.split(" ")
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

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors, id_tensors, shouldEnd


# This function is used to get the word embedding of current words from
def load_embed(word_vocab):
    # model = gensim.models.Word2Vec.load("./checkpoint/model1")
    model = KeyedVectors.load(FLAGS.emb_path, mmap='r')
    # f = codecs.open("./oov_list_mm", "w", "utf-8")
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
    for item, index in vocab_sorted:
        # print("space:", item,index, ".")
        n_sum += 1
        if item not in model:
            # f.write(item)
            # f.write("\n")
            i += 1
            if item not in w_dict.keys():
                w_dict[item] = 1
            else:
                w_dict[item] += 1
            # embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
            embed_list.append(np.random.uniform(-0.25, 0.25, 100).round(6).tolist())

            # embed_list.append(uni_list)
        else:
            # print(list(model[item]))
            embed_list.append(list(model[item]))
    print("no in:", i)
    print("all:", n_sum)
    # f.close()
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


# This function is used to get the word embedding of current words from
def get_embed(word_vocab):
    #model = gensim.models.Word2Vec.load("./checkpoint/model1")
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    f = codecs.open("./oov_list_mm", "w", "utf-8")
    #print("space: ", model[" "])
    embed_list = []
    #uni_list = [0 for i in range(150)]
    uni_list = [0 for i in range(100)]
    embed_list.append(uni_list)
    #print (embed_list)
    vocab_dict = word_vocab._token2index
    vocab_sorted = sorted(vocab_dict.items(),key=lambda asd:asd[1], reverse=False)
    i = 0
    n_sum = 0
    w_dict = {}
    for item, index in vocab_sorted:
        #print("space:", item,index, ".")
        n_sum += 1
        if item not in model:
            f.write(item)
            f.write("\n")
            i += 1
            if item not in w_dict.keys():
                w_dict[item] = 1
            else:
                w_dict[item] += 1
            #embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
            embed_list.append(np.random.uniform(-0.25, 0.25, 100).round(6).tolist())
           
            #embed_list.append(uni_list)
        else:
            #print(list(model[item]))
            embed_list.append(list(model[item]))
    print("no in:", i)
    print("all:", n_sum)
    f.close()
    #print(embed_list[0])
    embedding_array = np.array(embed_list, np.float32)
    print("no in:", i)
    print("all:", n_sum)
    f = codecs.open("./fre_sum", "w", "utf-8")
    for key, value in w_dict.items():
        f.write(key)
        f.write("\t")
        f.write(str(value))
        f.write("\n")
    f.close()
    return embedding_array


class DataReader:

    def __init__(self, word_tensor, label_tensor, batch_size):

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
        print(word_tensor.shape)

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        #print(x_batches.shape)
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        #print(x_batches.shape)
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        #print(len(self._x_batches))
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        #print (self.length)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y

    def __len__(self):

        return len(self._x_batches)


class DataReader_v2:

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
        # print(word_tensor.shape)

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



if __name__ == '__main__':

    vocab, word_tensors, max_length, label_tensors = load_data('data/demo', 1292, 10)

    count = 0
    for x, y in DataReader(word_tensors['valid'], label_tensors['valid'], 6).iter():
        count += 1
        print (x.shape, y.shape)
        if count > 0:
            break

