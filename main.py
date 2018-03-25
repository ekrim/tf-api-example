from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf


def input_fn_gen(is_training=False):
  def input_fn(params):
    file_list = glob.glob("data/*.tfrecords")
    dataset = tf.data.Dataset.list_files(file_list)
  
    if is_training:
      dataset = dataset.shuffle(1024).repeat()

    def process_dataset(file_name):
      dataset = tf.data.TFRecordDataset(file_name, buffer_size=8*1024*1024)
      return dataset

    dataset = dataset.interleave()

    #dataset = dataset.apply(
    #  tf.contrib.data.parallel_interleave(
    #  process_dataset, cycle_length=4, sloppy=True))
    
    dataset = dataset.shuffle(1024).prefetch(params["batch_size"])
    return dataset

  return input_fn


if __name__=="__main__":
  input_fn_gen({"batch_size":64})
