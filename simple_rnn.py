from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import time
import logging
import keras

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
# import seq2seq_model
import tensorflow.contrib.seq2seq as seq2seq


import random
import sys
from keras.preprocessing import sequence
#Taken from tensorflow tutorial

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
FLAGS = tf.app.flags.FLAGS
EOS = -1

_buckets = [(5, 5), (10, 10), (20, 20), (40, 40)]

def read_data(source_file, target_file):
    data_set = [[] for _ in _buckets]
    features=[]
    with open(source_file) as source_f:
        with open(target_file) as target_f:
            source, target = source.read_line(), target.read_line()
            while source and target: 
                source_ids = map(int,source.split(' '))
                target_ids = map(int,target.split(' '))
                target_ids.append(EOS)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                    source, target = source_file.readline(), target_file.readline()
    return data_set

print('Simple LSTM model')

print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))


# with tf.Session() as sess:
#     train_set = read_data('2p.train.onehot','train.seq2seq.txt.onehot')
#     train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
#     train_total_size = float(sum(train_bucket_sizes))

#     train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
#                            for i in xrange(len(train_bucket_sizes))]


#     # This is the training loop.
#     step_time, loss = 0.0, 0.0
#     current_step = 0
#     previous_losses = []
#     while True:
#         # Choose a bucket according to data distribution. We pick a random number
#         # in [0, 1] and use the corresponding interval in train_buckets_scale.
#         random_number_01 = np.random.random_sample()
#         bucket_id = min([i for i in xrange(len(train_buckets_scale))
#                          if train_buckets_scale[i] > random_number_01])


#         start_time = time.time()
#         encoder_inputs, decoder_inputs, target_weights = model.get_batch(
#             train_set, bucket_id)
#         _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
#                                      target_weights, bucket_id, False)
#         step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
#         loss += step_loss / FLAGS.steps_per_checkpoint
#         current_step += 1


#         if current_step % 5 == 0:
#             # Print statistics for the previous epoch.
#             perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
#             print ("global step %d learning rate %.4f step-time %.2f perplexity "
#                    "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
#                              step_time, perplexity))
        

#         if current_step%100 == 0: break










    #         labels=[]
        # with open('train.seq2seq.txt.onehot') as file:
        #     for line in file.readlines():
        #         labels.append(map(int,line.split(' ')))
        

        # #Both features and labels have long sentences. 
        # print(len(features), len(labels))
        # #Pad featues and labels to length of 40

        # x_train  = keras.preprocessing.sequence.pad_sequences(features, maxlen=max_len)
        # y_train  = keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_len)

        # print(x_train[0], x_train.shape)
        # print(y_train[0], y_train.shape)

