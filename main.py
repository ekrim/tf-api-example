from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import data_pipeline

def model_fn(features, labels, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  pass

param_dict = {
  'feature_columns':
  'n_classes':
  

classifier = tf.estimator.Estimator(
  model_fn=model_fn,
  params={
    'feature_columns':
    'n_class

if __name__=="__main__":
  input_fn = data_pipeline.input_fn_gen()

  sess = tf.Session()
  image, label = input_fn({"batch_size":64})

  for i in range(10):
    output = sess.run(label)
    print(output.shape)

    output = sess.run(image)
    print(print(output.shape))
  

