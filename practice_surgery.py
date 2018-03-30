from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def make_and_save():
  x = tf.placeholder(tf.float32, (None, 3), 'input')
  x = tf.layers.dense(x, 4)
  output = tf.identity(x, name="output")

  with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, "saved/model-checkpoint")

def restore():
  with tf.Session() as sess, tf.Graph().as_default() as g1:
    saver = tf.train.import_meta_graph("saved/model-checkpoint.meta")
    saver.restore(sess, "saved/model-checkpoint")
    g1_def = g1.as_graph_def()
    
  with tf.Session() as sess, tf.Graph().as_default() as g2:
    x_new = tf.placeholder(tf.float32, (None, 3), 'input')
    output = tf.identity(x_new, name="output")
    g2_def = g2.as_graph_def()

  with tf.Session() as sess, tf.Graph().as_default() as g_new:
    x = tf.placeholder(tf.float32, (None, 3), name="")
    y, = tf.import_graph_def(
      g2_def, 
      input_map={"input:0": x},
      return_elements=["output:0"])

    z, = tf.import_graph_def(
      g1_def, 
      input_map={"input:0": y},
      return_elements=["output:0"])
    

    print([op.name for op in g_new.get_operations()])

if __name__=='__main__':
  restore()

  
