from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf


def dataset_parser(value):
  keys_to_features = {
    'image': tf.FixedLenFeature((), tf.string, ''),
    'label': tf.FixedLenFeature((), tf.int64, -1)}
  parsed = tf.parse_single_example(value, keys_to_features)
  
  image = tf.image.decode_image(tf.reshape(parsed['image'], shape=[]), 3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  
  label = tf.cast(tf.reshape(parsed['label'], shape=[]), dtype=tf.int32) - 1
  return image, label
  
def input_fn_gen(is_training=False):
  def input_fn(params):
    file_list = glob.glob("data/*.tfrecords")
    dataset = tf.data.Dataset.list_files(file_list)
  
    if is_training:
      dataset = dataset.shuffle(1024).repeat()
    
    def process_dataset(file_name):
      dataset = tf.data.TFRecordDataset(file_name, buffer_size=8*1024*1024)
      return dataset

    dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
      process_dataset, cycle_length=4, sloppy=True))
    
    dataset = dataset.shuffle(1024)
    dataset = dataset.map(dataset_parser, num_parallel_calls=64)
    dataset = dataset.prefetch(params["batch_size"])
  
    image, label = dataset.make_one_shot_iterator().get_next()
    return image, label 

  return input_fn


if __name__=="__main__":
  input_fn = input_fn_gen()

  sess = tf.Session()
  image, label = input_fn({"batch_size":64})

  output = sess.run(label)
  print(output)
  print(type(output))
  print(output.shape)
