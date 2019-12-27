#!/usr/bin/env python
#coding:utf8
import numpy
import argparse
import logging
import os
import sys
#import cPickle as pkl
#from helper import Config
#from helper import Dataset
#from helper import DataLoader
#from helper import prepare_data
#from helper import test
import data_reader as dr
import datetime
from datetime import timedelta
import codecs
import time
import tensorflow as tf
from sklearn import metrics
from model import SummaRuNNer

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

FLAGS=tf.app.flags.FLAGS
# 批量预测
tf.app.flags.DEFINE_boolean("predict_incrementally",True,"if need to predict only the latest part") #是否需要增量预测
tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/dp_struccontent_summary_model/test","predict result")
if FLAGS.predict_incrementally == True:
    tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr_test.db/temp_summary_high_ctr_seg","path of traning data.")  # recreason_before_lm_model_incre
if FLAGS.predict_incrementally == False:
    tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr_test.db/temp_summary_high_ctr_seg","path of traning data.")  # TODO 用的是test库
#tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_lm/test","predict result")
#tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_bak","path of traning data.")  # bert数据过滤
tf.app.flags.DEFINE_string("ckpt_dir","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/","checkpoint location for the model")
# tf.app.flags.DEFINE_string("vocabulary_word2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/word2index1203_2.pkl","vocabulary_word2index")
# tf.app.flags.DEFINE_string("vocabulary_label2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/label2index1203_2.pkl","vocabulary_label2index")
tf.app.flags.DEFINE_string("emb_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/model2","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_integer("workers", 1, "work node num")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config_proto.gpu_options.allow_growth = True
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



args = parser.parse_args()
max_sen_length = args.sen_len
max_doc_length = args.doc_len
tart_time = time.time()
logging.info('generate config')

start = datetime.datetime.now()
print("starting time : " + str(start))
workers = FLAGS.workers
list_name = tf.gfile.ListDirectory(FLAGS.training_data_path)
total_file_num = len(list_name)
print("list_name : " + str(list_name))
print("taskindex : " + str(FLAGS.task_index))
print("total file num : " + str(total_file_num))
cur_file_names = list_name[FLAGS.task_index:total_file_num:FLAGS.total_workers]
print("cur_file_names : " + str(cur_file_names))
fileList = [os.path.join(FLAGS.training_data_path, a) for a in cur_file_names]
print("fileList : " + str(fileList))

batch_size = 1
time1 = time.time()
write_index = 0
shouldEnd = False
sub_task_id = 0

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        # saver = tf.compat.v1.train.import_meta_graph('./runs/1564993889/checkpoints/model-512.meta')
        # module_file = tf.train.latest_checkpoint("./runs/1564993889/" + 'checkpoints/')
        saver = tf.compat.v1.train.import_meta_graph(FLAGS.ckpt_dir + 'model.meta')
        module_file = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        saver.restore(sess, module_file)
        # f0 = codecs.open("summaries", "w", "utf-8")

        while not shouldEnd:
            input_x = graph.get_operation_by_name("inputs/x_input").outputs[0]
            predict = graph.get_operation_by_name("score_layer/prediction").outputs[0]
            resultlines = []
            loss_sum = 0
            count = 0
            # word_vocab, word_tensors, max_doc_length, label_tensors = \
            #     dr.load_test_v2(args.train_file, max_doc_length, max_sen_length)
            # word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors = \
            #     dr.load_test_v3(args.train_file, max_doc_length, max_sen_length)
            word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors, shouldEnd = \
                dr.load_from_fileList(fileList, write_index, max_doc_length, max_sen_length)
            # test_reader = dr.DataReader(word_tensors['test'], label_tensors['test'],
            #                             batch_size)
            test_reader = dr.DataReader_v2(word_tensors['test'], label_tensors['test'],
                                       id_tensors['test'], batch_size)

            for x, y, z in test_reader.iter():
                count += 1
                x = x[0]
                y = y[0]
                # print (x)
                y_ = sess.run(predict, feed_dict = {input_x : x})
                # ys_ = sess.run(predict, feed_dict = {input_x, xs})
                # for x, y, y_ in zip(xs[0], ys[0], ys_[0]):
                max_len = 0
                for i, item in enumerate(x):
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

                tmp_str = ''
                actual_length = 0
                index = 0
                out_flag = 0
                top_sentence_index = numpy.argmax(y_)
                # print(top_sentence_index)
                # print(y_)
                while len(tmp_str) < 10 and out_flag == 0:
                    y_[top_sentence_index] = 1
                    for word in x[top_sentence_index]:
                        if word == 1:
                            # tmp_str += str(y_[top_sentence_index]) + '\t' + str(y[top_sentence_index]) + '\t'
                            continue
                        elif word == 2:
                            # tmp_str += '\n'
                            break
                        else:
                            tmp_str += str(word_vocab.token(word))
                    if top_sentence_index == len(x) - 1:
                        out_flag = 1
                        # tmp_str += '\n'
                    elif len(tmp_str) < 10:
                        top_sentence_index += 1
                    else:
                        out_flag = 1
                resultlines.append(str(z)[1:-1] + FLAGS.delimiter + tmp_str + "\n")
                # f0.write(str(z)[1:-1] + '\t' + tmp_str + '\n')
            # print(count)
            # print(sub_task_id)
            result_filename = FLAGS.predict_target_file + "_" + str(FLAGS.task_index) + "_" + str(sub_task_id)
            if FLAGS.predict_incrementally == True:
                result_filename = result_filename + "_" + str(datetime.date.today())
            predict_target_file_f = tf.gfile.GFile(result_filename, 'w')
            for result in resultlines:
                predict_target_file_f.write(result)
            predict_target_file_f.close()
            write_index = write_index + FLAGS.line_per_file
            sub_task_id = sub_task_id + 1

        time_dif = timedelta(seconds=int(round(time.time() - time1)))
        print("Time usage:", time_dif)
