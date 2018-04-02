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
  input_fn = data_pipeline.input_fn_gen(mode='train', batch_size=64)

  with tf.Session() as sess:
  
    classifier = tf.estimator.Estimator(
      model_fn=models.model_fn)
      #model_dir="/tmp/example_model") 
      
    classifier.train(
      input_fn=input_fn,
      steps=20)
      
