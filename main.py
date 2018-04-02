from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import data_pipeline
import models


if __name__=="__main__":
  cifar = data_pipeline.Cifar10Input()
  train_input_fn = cifar.input_fn_factory(mode='train', batch_size=64)
  eval_input_fn = cifar.input_fn_factory(mode='validate', batch_size=64)

  with tf.Session() as sess:
  
    classifier = tf.estimator.Estimator(
      model_fn=models.model_fn)
      #model_dir="/tmp/example_model") 
      
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
      
