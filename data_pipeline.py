from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


HEIGHT = 32
WIDTH = 32
DEPTH = 3


def dataset_parser(value):
  keys_to_features = {
    'image': tf.FixedLenFeature((), tf.string, ''),
    'label': tf.FixedLenFeature((), tf.int64, -1)}
  parsed = tf.parse_single_example(value, features=keys_to_features)
  image = tf.decode_raw(parsed['image'], tf.uint8)
  image.set_shape([HEIGHT*WIDTH*DEPTH])
  
  image = tf.cast(
    tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1,2,0]),
    tf.float32)
  
  label = tf.cast(parsed['label'], tf.int32)
  
  return image, label

  
def input_fn_gen(mode="test"):
  def input_fn(params):
    file_list = glob.glob("data/*.tfrecords")
    dataset = tf.data.Dataset.list_files(file_list)
  
    if mode=="train":
      dataset = dataset.shuffle(buffer_size=min(len(file_list), 1024)).repeat()
    
    def process_dataset(file_name):
      dataset = tf.data.TFRecordDataset(file_name, buffer_size=8*1024*1024)
      return dataset

    dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
      process_dataset, cycle_length=4, sloppy=True))
    
    if mode=="train":
      dataset = dataset.shuffle(1024)

    dataset = dataset.map(dataset_parser, num_parallel_calls=64)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params["batch_size"]))
    dataset = dataset.prefetch(params["batch_size"])
  
    image, label = dataset.make_one_shot_iterator().get_next()
    return image, label 

  return input_fn


if __name__=="__main__":
  input_fn = input_fn_gen()

  sess = tf.Session()
  image, label = input_fn({"batch_size":64})

  for i in range(10):
    output = sess.run(label)
    print(output.shape)

    output = sess.run(image)
    print(print(output.shape))
  

