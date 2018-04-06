from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf
 

def model_test(features, mode):
  '''A simple model, should get about 65% accuracy on CIFAR-10
  '''
  tf.summary.image('images', features['image'], max_outputs=3)

  conv1 = tf.layers.conv2d(
    inputs=features['image'],
    filters=96,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  pool1 = tf.layers.average_pooling2d(
    inputs=conv1,
    pool_size=[3,3],
    strides=3)

  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=32,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  pool2 = tf.layers.average_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2)

  flat1 = x = tf.reshape(pool2, [-1, 32*4*4])
  
  for i in range(10):
    x = tf.layers.dense(inputs=x, units=144)
  
  logits = tf.layers.dense(inputs=x, units=10)

  return logits 


def model_all_cnn_c(features, mode):
  '''The all convolutional net ALL-CNN-C
  https://arxiv.org/abs/1412.6806
  '''
  tf.summary.image('images', features['image'], max_outputs=1)

  use_dropout = mode == tf.estimator.ModeKeys.TRAIN

  drop1 = tf.layers.dropout(features['image'], rate=0.2, training=use_dropout)

  conv1 = tf.layers.conv2d(
    inputs=drop1,
    filters=96,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)
  
  conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=96,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=96,
    kernel_size=[3,3],
    padding="valid",
    strides=(2,2),
    activation=tf.nn.relu)

  drop2 = tf.layers.dropout(conv3, rate=0.5, training=use_dropout)

  conv4 = tf.layers.conv2d(
    inputs=drop2,
    filters=192,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  conv5 = tf.layers.conv2d(
    inputs=conv4,
    filters=192,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  conv6 = tf.layers.conv2d(
    inputs=conv5,
    filters=192,
    kernel_size=[3,3],
    padding="valid",
    strides=(2,2),
    activation=tf.nn.relu)

  drop3 = tf.layers.dropout(conv6, rate=0.5, training=use_dropout)

  conv7 = tf.layers.conv2d(
    inputs=drop3,
    filters=192,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)

  conv8 = tf.layers.conv2d(
    inputs=conv7,
    filters=192,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.relu)

  conv9 = tf.layers.conv2d(
    inputs=conv8,
    filters=10,
    kernel_size=[1,1],
    padding="valid",
    activation=tf.nn.relu)

  logits = tf.reduce_mean(conv9, [1,2])  
  return logits
  

def model_fn_closure(model_name='test'):
  '''model_name is one of "test" or "all_cnn"
  '''
  if model_name == 'test':
    inference_fn = model_test
  elif model_name == 'all_cnn':
    inference_fn = model_all_cnn_c  
  else:
    assert False, 'model not implemented'

  def model_fn(features, labels, mode, params):
    '''Model function for estimators
    '''
    logits = inference_fn(features, mode)
  
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
  
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) 
     
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

      training_hooks = [tf.train.SummarySaverHook(
        save_steps=50,
        summary_op=tf.summary.merge_all())]        

      return tf.estimator.EstimatorSpec(
        mode=mode, 
        loss=loss, 
        train_op=train_op,
        training_hooks=training_hooks)
  
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metric_ops = {"accuracy": accuracy}
  
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss, 
      eval_metric_ops=eval_metric_ops)

  return model_fn


if __name__=="__main__":
  features = {
    "image": tf.placeholder(tf.float32, (None, 32, 32, 3))}
  labels = tf.placeholder(tf.int32, (None, 1))

  model_fn_closure('test')(features, labels, tf.estimator.ModeKeys.TRAIN, {})
