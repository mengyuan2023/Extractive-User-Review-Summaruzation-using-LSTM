#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
# from model import SummaRuNNer
from model_v2 import SummaRuNNer
import codecs
#import data_input_helper as data_helpers
#from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import data_reader as dr
import data_reader_v2 as dr2
#import tensorlayer as tl
import logging
# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS
# 批量预测
# tf.app.flags.DEFINE_boolean("predict_incrementally",True,"if need to predict only the latest part") #是否需要增量预测
# tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/recreason_lm_result_new/test","predict result")
# if FLAGS.predict_incrementally == True:
#     tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/recreason_before_lm_model_incre","path of traning data.")  # recreason_before_lm_model_incre
# if FLAGS.predict_incrementally == False:
#     tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/recreason_before_lm_model_all","path of traning data.")  # TODO 用的是test库
# #tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_lm/test","predict result")
# #tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_bak","path of traning data.")  # bert数据过滤
# tf.app.flags.DEFINE_string("ckpt_dir","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/","checkpoint location for the model")
# tf.app.flags.DEFINE_string("vocabulary_word2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/word2index1203_2.pkl","vocabulary_word2index")
# tf.app.flags.DEFINE_string("vocabulary_label2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/label2index1203_2.pkl","vocabulary_label2index")
# tf.app.flags.DEFINE_string("emb_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/classify_emb_1203.txt","word2vec's vocabulary and vectors")
#
# config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
# config_proto.gpu_options.allow_growth = True


# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("train_data_file", "/var/proj/sentiment_analysis/data/cutclean_tiny_stopword_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "./data/split_90", "Data source for the positive data.")
#tf.flags.DEFINE_string("valid_data_file", "../data/split90/valid", "Data source for the positive data.")
#tf.flags.DEFINE_string("test_data_file", "../data/split90/test", "Data source for the positive data.")
#tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
#需要修改
tf.flags.DEFINE_string("w2v_file", "../model.0201", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train(w2v_model):
    # Training
    # ==================================================
    max_sen_length = 20
    max_doc_length = 60
 #    word_vocab, word_tensors, max_doc_length, label_tensors = \
 # \
 #        dr.load_data_excel_v2(FLAGS.train_data_file, max_doc_length, max_sen_length)

    word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors = \
 \
        dr2.load_data_excel(FLAGS.train_data_file, max_doc_length, max_sen_length)

    # train_reader = dr.DataReader(word_tensors['train'], label_tensors['train'], FLAGS.batch_size)

    # valid_reader = dr.DataReader(word_tensors['valid'], label_tensors['valid'], FLAGS.batch_size)

    # test_reader = dr.DataReader(word_tensors['test'], label_tensors['test'], 1)

    train_reader = dr2.DataReader(word_tensors['train'], label_tensors['train'], id_tensors['train'], FLAGS.batch_size)

    valid_reader = dr2.DataReader(word_tensors['valid'], label_tensors['valid'], id_tensors['valid'], FLAGS.batch_size)

    test_reader = dr2.DataReader(word_tensors['test'], label_tensors['test'], id_tensors['test'], 1)

    # pretrained_embedding = dr.get_embed(word_vocab)
    print("start get embeddings : " + str(datetime.datetime.now()))
    pretrained_embedding = dr2.get_embed(word_vocab)
    print("start training : " + str(datetime.datetime.now()))
    #embedding_size = 150
    embedding_size = 100
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print(word_vocab.size)
            Summa = SummaRuNNer(
                word_vocab.size, embedding_size, FLAGS.batch_size, pretrained_embedding
            )

            #loss_sum = tf.Variable(initial_value=0, dtype=tf.float32)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_params = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(Summa.loss, var_list=train_params)
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            batches = train_reader
            valid_batches = valid_reader
            sess.run(tf.global_variables_initializer())
            #step = 0
            min_eval_loss = float('Inf')
            f = codecs.open(timestamp + "_note", "w", "utf-8")
            f.write(str(FLAGS) + '\n')
            for epoch in range(FLAGS.num_epochs):
                step = 0
                loss_sum = 0
                #print(len(batches))
                for x_batch, y_batch, z_batch in batches.iter():
                    step += 1
                    # feed_dict = {
                    #   Summa.x: x_batch[0],
                    #   Summa.y: y_batch[0],
                    # }
                    feed_dict = {
                      Summa.x: x_batch,
                      Summa.y: y_batch,
                    }
                    sess.run(train_op,feed_dict)
                    loss = sess.run(
                        [Summa.loss],
                        feed_dict)
                    predict = sess.run([Summa.y_], feed_dict)
                    loss_sum += loss[0]
                    if step % 128 == 0 and step != 0:
                        print (str(datetime.datetime.now()) + 'Epoch ' + str(epoch) + ' Loss: ' + str(loss_sum / 128.0))
                        f.write(str(datetime.datetime.now()) + 'Epoch ' + str(epoch) + ' Loss: ' + str(loss_sum / 128.0) + '\n')
                        loss_sum = 0
                    if step % 512 == 0 and step != 0:
                        eval_loss = 0
                        for x_batch, y_batch, z_batch in valid_batches.iter():
                            # feed_dict = {
                            #     Summa.x: x_batch[0],
                            #     Summa.y: y_batch[0],
                            # }
                            feed_dict = {
                                Summa.x: x_batch,
                                Summa.y: y_batch,
                            }
                            loss = sess.run(
                                [Summa.loss],
                                feed_dict)
                            eval_loss += loss[0]
                        print (str(datetime.datetime.now()) + 'epoch ' + str(epoch) + ' Loss in validation: ' + str(
                                eval_loss * 1.0 / valid_reader.length))
                        f.write(str(datetime.datetime.now()) + 'epoch ' + str(epoch) + ' Loss in validation: ' + str(
                            eval_loss * 1.0 / valid_reader.length) + '\n')
                        if eval_loss < min_eval_loss:
                            min_eval_loss = eval_loss

                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                            f.write("Saved model checkpoint to {}\n".format(path) + '\n')



if __name__ == "__main__":
    train("model.0201")


