#!/usr/bin/env python
#coding:utf8
import numpy
import argparse
import logging
import sys
#import cPickle as pkl
#from helper import Config
#from helper import Dataset
#from helper import DataLoader
#from helper import prepare_data
#from helper import test
import data_reader_v2 as dr2
import datetime
import codecs
import time
import tensorflow as tf
from sklearn import metrics
from model import SummaRuNNer
from decimal import getcontext, Decimal

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--sen_len', type=int, default=100)
parser.add_argument('--doc_len', type=int, default=100)
parser.add_argument('--train_file', type=str, default='./data/split_90/')
parser.add_argument('--validation_file', type=str, default='./data/split_90/valid')
# parser.add_argument('--model_dir', type=str, default='./runs/1532436443/checkpoints/')
parser.add_argument('--model_dir', type=str, default='./checkpoints/')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--hidden', type=int, default=110)
parser.add_argument('--lr', type=float, default=1e-4)

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
print("start : " + str(datetime.datetime.now()))
FLAGS = tf.flags.FLAGS

args = parser.parse_args()
print("dc", args.doc_len)
# max_sen_length = args.sen_len
# max_doc_length = args.doc_len

max_sen_length = 20
max_doc_length = 60

logging.info('generate config')
word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors = \
    dr2.load_test(args.train_file, max_doc_length, max_sen_length)

batch_size = 1
time1 = time.time()
test_reader = dr2.DataReader(word_tensors['test'], label_tensors['test'], id_tensors['test'],
                         batch_size)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        # saver = tf.compat.v1.train.import_meta_graph('./runs/1564993889/checkpoints/model-512.meta')
        # module_file = tf.train.latest_checkpoint("./runs/1564993889/" + 'checkpoints/')
        # saver = tf.compat.v1.train.import_meta_graph('./runs/1564996050/checkpoints/model-512.meta')
        # module_file = tf.train.latest_checkpoint("./runs/1564996050/" + 'checkpoints/')
        # saver = tf.compat.v1.train.import_meta_graph('./runs/1565852057/checkpoints/model-3072.meta')
        # module_file = tf.train.latest_checkpoint("./runs/1565852057/" + 'checkpoints/')
        saver = tf.compat.v1.train.import_meta_graph('./runs/1565927208/checkpoints/model-1536.meta')
        module_file = tf.train.latest_checkpoint("./runs/1565927208/" + 'checkpoints/')
        saver.restore(sess, module_file)
        input_x =  graph.get_operation_by_name("inputs/x_input").outputs[0]
        predict =  graph.get_operation_by_name("score_layer/prediction").outputs[0]
        # f = codecs.open(args.model_dir+"scores" , "w", "utf-8")
        f0 = codecs.open("summaries", "w", "utf-8")
        f = codecs.open("scores", "w", "utf-8")
        f1 = codecs.open("test_res", "w", "utf-8")
        f2 = codecs.open("test_res_pos", "w", "utf-8")
        f3 = codecs.open("test_res_neg", "w", "utf-8")
        jk = 0
        loss_sum = 0
        acc_sentence = 0
        acc_doc = 0
        doc_count = 0
        test_y = []
        predict_prob_y = []
        count = 0
        doc_empty_count = 0
        for x, y, z in test_reader.iter():
            count += 1
            x = x[0]
            y = y[0]
            #print (x)
            y_ = sess.run(predict, feed_dict = {input_x: x})
            flag = 0
            max_len = 0
            for i,item in enumerate(x):
                #print item
                temp = 0
                for sub_item in item:
                    #print(type(int(sub_item)))
                    if sub_item > 0:
                        temp += 1
                        #print temp
                if temp == 0:
                    x = x[:i, :max_len]
                    y_ = y_[:i]
                    y = y[:i]
                    break
                if temp > max_len:
                    max_len = temp
            x = x[:, :max_len]

            #print(max_len, len(x), len(y), len(y_))
            # quit()
            flag = 1
            tmp_str = ''
            actual_length = 0
            index = 0
            out_flag = 0
            top_sentence_index = numpy.argmax(y_)
            left = top_sentence_index - 1
            right = top_sentence_index + 1
            res = {}
            f.write(str(z)[1:-1])
            f.write('\t')
            for score in y_:
                score = '%.4f'%score
                f.write(str(score))
                f.write(' ')
            f.write('\n')
            # res.append(top_sentence_index)
            # print(top_sentence_index)
            # print(y_)
            while len(tmp_str) < 10 and out_flag == 0:
                # print(top_sentence_index, len(x))
                y_[top_sentence_index] = 2
                cur_sentence = ''
                tmp_str += y[top_sentence_index]
                # for word in x[top_sentence_index]:
                #     if word == 1:
                #         # tmp_str += str(y_[top_sentence_index]) + '\t' + str(y[top_sentence_index]) + '\t'
                #         continue
                #     elif word == 2:
                #         # tmp_str += '\n'
                #         break
                #     else:
                #         cur_sentence += str(word_vocab.token(word))
                #         tmp_str += str(word_vocab.token(word))
                res[top_sentence_index] = y[top_sentence_index]
                # if top_sentence_index == len(x) - 1:
                #     out_flag = 1
                #     tmp_str += '\n'
                if len(tmp_str) < 10:
                    if left < 0 and right > len(x) - 1:
                        out_flag = 1
                    elif left < 0:
                        top_sentence_index = right
                        # res.append(right)
                        right += 1
                    elif right > len(x) - 1:
                        top_sentence_index = left
                        left -= 1
                        # res.append(left)
                    else:
                        if y_[right] >= y_[left]:
                            top_sentence_index = right
                            right += 1
                            # res.append(right)
                        else:
                            top_sentence_index = left
                            left -= 1
                            # res.append(left)
                    # top_sentence_index += 1
                else:
                    out_flag = 1
                    # tmp_str += '\n'
            tmp_str = ''
            for i, score in enumerate(y_):
                if score == 2:
                    try:
                        tmp_str += res[i]
                    except KeyError:
                        print(z)
                        print(i)
                        print(tmp_str)
                        print(x)
                        print(res)
                        print(y)
            tmp_str += '\n'
                # if out_flag == 0:
                #     print(tmp_str)
            f0.write(str(z)[1:-1] + '\t' + tmp_str)
            if count % 1000 == 0:
                print(str(count) +  ' ' + str(datetime.datetime.now()))
        #     tmp_str = ''
        #     doc_flag = 0
        #     for i, (sentence, score, t) in enumerate(zip(x, y_, y)):
        #         if score != 1:
        #             score = 0
        # 
        #     # if len(tmp_str) > 20:
        #     #     print(len(tmp_str))
        # 
        #     # tmp_str = ''
        #     # for sentence, score, t in zip(x, y_, y):
        #     #     #print(score,t)
        #     #     #print(type(score.float))
        #     #     #score = score.float
        #     #     f.write(str(score))
        #     #     f.write(" ")
        #         test_y.append(t)
        #         predict_prob_y.append(score)
        #     #     if score < 0.4:
        #     #         score = 0
        #     #     else:
        #     #         score = 1
        #     #     #predict_prob_y.append(score)
        #         if score == t:
        #             acc_sentence += 1
        #             for word in sentence:
        #                 # tmp_str += str(word_vocab.token(word))
        #                 if word == 1:
        #                     tmp_str += str(score) + '\t' + str(t) + '\t'
        #                 elif word == 2:
        #                     tmp_str += "\n"
        #                     break
        #                 else:
        #                     tmp_str += str(word_vocab.token(word))
        #         else:
        #             flag = 0
        #             for word in sentence:
        #                 # tmp_str += str(word_vocab.token(word))
        #                 if word == 1:
        #                     tmp_str += str(score) + '\t' + str(t) + '\t'
        #                 elif word == 2:
        #                     tmp_str += "\n"
        #                     break
        #                 else:
        #                     tmp_str += str(word_vocab.token(word))
        #         jk += 1
        #     # f1.write(tmp_str + '\n')
        #     #     print("jk:", jk)
        #     if flag == 1:
        #         acc_doc += 1
        #     #     f2.write(tmp_str)
        #     #     f2.write("\n")
        #     # else:
        #     #     f3.write(tmp_str)
        #     #     f3.write("\n")
        #     if 1 not in y:
        #         # print(tmp_str)
        #         doc_empty_count += 1
        #     doc_count += 1
        # #     # f.write("\n")
        # test_auc = metrics.roc_auc_score(test_y, predict_prob_y)
        # print(test_auc)
        # print(count)
        # print(doc_empty_count)
        # print(acc_sentence, jk)
        # print(acc_doc, doc_count)
        # f1.write("auc: " + str(test_auc) + " acc_sen: " + str(acc_sentence) + " total sen: " + str(jk) + " acc_doc: " + str(acc_doc) + " total doc: " + str(doc_count))
        f.close()
print("end : " + str(datetime.datetime.now()))