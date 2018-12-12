#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import tempfile
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorboard import summary as summary_lib

from movieReviewData import movieReviewData
import argparse


tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

#################### Define some parameter here ###############################

tf.app.flags.DEFINE_string('model_dir', './models/ckpt/', 'Dir to save a model and checkpoints')
tf.app.flags.DEFINE_string('saved_dir', './models/pb/', 'Dir to save a model for TF serving')
#tf.app.flags.DEFINE_string('step_size', '20000', 'Step Size')
tf.app.flags.DEFINE_string('step_size', '10', 'Step Size')
#tf.app.flags.DEFINE_string('batch_size', './models/pb/', 'Batch Size')
FLAGS = tf.app.flags.FLAGS


#parser = argparse.ArgumentParser()
#parser.add_argument('--model_dir', default= './models/ckpt/', type=str, help='Dir to save a model and checkpoints') 
#parser.add_argument('--saved_dir', default='./models/pb/', type=str, help='Dir to save a model for TF serving')
#parser.add_argument('--step_size', default=10, type=int, help='Step size')
#parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
#args = parser.parse_args()

#num_pid = int(args.size)
#proc = int(args.proc)


#print("================== Custom model in LSTM =========================")
#print(parser.print_help())
#print("=================================================================")
###############################################################################
# Load data
data = movieReviewData()

# Preprocessing data
data.preProcessing()

vocab_size = data.vocab_size
embedding_size = 50
sentence_size = data.sentence_size
#start_id = 1
#oov_id = 2
#index_offset = 2

#vocab_size = 5000
#embedding_size = 50
#sentence_size=200
#start_id = 1
#oov_id = 2
#index_offset = 2

#    MAX_SIZE = 1000
#MAX_SIZE = 25000

#     Prepare data
x_train = data.x_train
x_test = data.x_test

y_train = data.y_train
y_test  = data.y_test


# Get length of each sentence
#x_len_train = np.array([min(len(x), sentence_size) for x in data.x_train_variable[:MAX_SIZE]])
#x_len_test = np.array([min(len(x), sentence_size) for x in data.x_test_variable[:MAX_SIZE]])
    

def LSTM_model_fn(features, labels, mode):
    """
        Description: Custom model LSTM
        Usage:
        return: 
    """
    
    # [batch_size x sentence_size x embedding_size]
    
    inputs = tf.contrib.layers.embed_sequence(
            features['x'],vocab_size,embed_dim=embedding_size,
            initializer=tf.random_uniform_initializer(-1.0,-1.0))
    
    # create an LSTM cell of size 100
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

    # Getting sequence length from features sucks -> initialize sequence length here
    sequence_length = tf.count_nonzero(features['x'], 1)
    
    # create the complete LSTM
#    _, final_states = tf.nn.dynamic_rnn(
#        lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)
    
    _, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length = sequence_length, dtype=tf.float32)
    
    # get the final hidden states of dimensionality [batch_size x sentence_size]
    outputs = final_states.h   
    logits = tf.layers.dense(inputs=outputs, units=1)
    
    if labels is not None: # vertical array
        labels = tf.reshape(labels, [-1, 1])
    
    # Compute loss.
#    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
#    loss = tf.losses.softmax_cross_entropy(labels,logits=logits)
   
#    loss = tf.losses.mean_squared_error(labels,logits)
    
    # Compute predictions.
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "next": tf.round(tf.sigmoid(logits)),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
      }

    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions["next"])
    
    loss = tf.losses.sigmoid_cross_entropy(labels,logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["next"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops) 
    
    
    
    
#lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)
#RNN_classifier.export_savedmodel(FLAGS.saved_dir, serving_input_receiver_fn=serving_input_receiver_fn)


#def parser(x, length, y):
#    '''
#        Description: 
#        Usage:
#    '''
#    
#    features = {"x": x, "len": length}
#    return features, y


def parser(x, y):
    '''
        Description: 
        Usage:
    '''
    
    features = {"x": x}
#    y_ = {"next":y}
    return features, y

#def train_input_fn(x_train,y_train,x_len_train,batch_size):
#    '''
#        Description: 
#        Usage:
#    '''
#    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train,x_len_train))
#    dataset = dataset.shuffle(1000).batch(batch_size).map(parser).repeat()
#    iterator = dataset.make_one_shot_iterator()
#    return iterator.get_next()

def train_input_fn(x_train,y_train,batch_size):
    '''
        Description: 
        Usage:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    dataset = dataset.shuffle(1000).batch(batch_size).map(parser).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

#x = x_train[0]  
#y = y_train[0]
#x_len = x_len_train[0]

#def eval_input_fn(x_train,y_train,x_len,batch_size):
#    '''
#        Description: 
#        Usage:
#    '''
#    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train,x_len))
#    dataset = dataset.batch(batch_size).map(parser)
#    iterator = dataset.make_one_shot_iterator()
#    return iterator.get_next()  

def eval_input_fn(x_train,y_train,batch_size):
    '''
        Description: 
        Usage:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    dataset = dataset.batch(batch_size).map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()  


def serving_input_receiver_fn():
    """
        Description: This is used to define inputs to serve the model.
        Usage:
        return: ServingInputReciever
        Ref: https://www.tensorflow.org/versions/r1.7/api_docs/python/tf/estimator/export/ServingInputReceiver
    """
    
#    reciever_tensors = {
#        # The size of input sentence is flexible.
#        "sentence":tf.placeholder(tf.int32, [None, 1]),
#        "len": tf.placeholder(tf.int32,[None,None])
#    }

#    reciever_tensors = {
#        # The size of input sentence is flexible.
#        "sentence":tf.placeholder(tf.int32, [None, 1])
#    }

    reciever_tensors = {
        # The size of input sentence is flexible.
        "sentence":tf.placeholder(tf.int32, [None, 1])
    }
    
    
    
#    labels = tf.reshape(reciever_tensors["sentence"], [200, 1])
    
#    length = tf.count_nonzero(reciever_tensors["sentence"],axis=1)
        
    features = {
        # Resize given images.
        "x": tf.reshape(reciever_tensors["sentence"], [200, 1])
    }

#    features = {
#        # Resize given images.
#        "x":reciever_tensors["sentence"],
#        "len": reciever_tensors["len"]
#    }
    
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)
def main(unused_argv):
    # Create the Estimator
    RNN_classifier = tf.estimator.Estimator(model_fn=LSTM_model_fn, model_dir= FLAGS.model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "sigmoid_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    STEP_SIZE = int(FLAGS.step_size)
#    STEP_SIZE = 20000
    
#    RNN_classifier.train(
#        input_fn=lambda: train_input_fn(x_train, x_len_train,y_train,batch_size=100),
#        steps=STEP_SIZE,
#        hooks=[logging_hook]) 
    
    RNN_classifier.train(
        input_fn= lambda: train_input_fn(x_train,y_train,batch_size=100),
        steps=STEP_SIZE,
        hooks=[logging_hook])  
    
#    a = train_input_fn(x_train, x_len_train,y_train,batch_size=100)
#    features=a[0]
#    labels=a[1]

    
#    eval_results = RNN_classifier.evaluate(
#       input_fn = lambda: eval_input_fn(x_test, x_len_test,y_test,batch_size=100))

    eval_results = RNN_classifier.evaluate(
       input_fn = lambda: eval_input_fn(x_test,y_test,batch_size=100))
    
    print(eval_results)
    
    # Save the model
    RNN_classifier.export_savedmodel(FLAGS.saved_dir, serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == "__main__":
    tf.app.run()