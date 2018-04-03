from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf


class Cifar10Input:

  def __init__(self):
    self.HEIGHT = 32
    self.WIDTH = 32
    self.DEPTH = 3
    
    self.file_list = {
      'train': ['data/data_batch_{:d}.tfrecords'.format(i) for i in range(1,5)],
      'validate': ['data/data_batch_5.tfrecords'],
      'test': ['data/test_batch.tfrecords']}

  def input_fn_factory(self, mode='test', batch_size=64):
    def input_fn():
      file_list = self.file_list[mode]
      dataset = tf.data.Dataset.list_files(file_list)
    
      if mode=='train':
        dataset = dataset.shuffle(buffer_size=min(len(file_list), 1024)).repeat()
      
      def process_tfrecord(file_name):
        dataset = tf.data.TFRecordDataset(file_name, buffer_size=8*1024*1024)
        return dataset

      dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
        process_tfrecord, cycle_length=4, sloppy=True))
      
      if mode=='train':
        dataset = dataset.shuffle(1024)

      dataset = dataset.map(self.parser_factory(mode), num_parallel_calls=64)
      dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

      dataset = dataset.prefetch(batch_size)
    
      image, label = dataset.make_one_shot_iterator().get_next()
      return {'image': image}, label 

    return input_fn
    
  def parser_factory(self, mode):
    def parser_fn(value):
      keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, ''),
        'label': tf.FixedLenFeature((), tf.int64, -1)}
      parsed = tf.parse_single_example(value, features=keys_to_features)
      image = tf.decode_raw(parsed['image'], tf.uint8)
      image.set_shape([self.HEIGHT * self.WIDTH * self.DEPTH])
      
      image = tf.cast(
        tf.transpose(tf.reshape(image, [self.DEPTH, self.HEIGHT, self.WIDTH]), [1,2,0]),
        tf.float32)
      
      if mode == 'train':
        image = self.preprocess_fn(image)
      
      image = tf.image.per_image_standardization(image)
      label = tf.cast(parsed['label'], tf.int32)
      return image, label

    return parser_fn

  def preprocess_fn(self, image):
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
    image = tf.image.random_flip_left_right(image)
    return image


if __name__=='__main__':
  cifar = Cifar10Input()
  input_fn = cifar.input_fn_factory(mode='train', batch_size=2)
  
  with tf.Session() as sess:
    image, label = input_fn()

    for i in range(10):
      images = sess.run(image['image'])
      print(images.shape)
      labels = sess.run(label)
      print(labels.shape)

