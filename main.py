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


def build_model():
  inp = x = tf.keras.layers.Input(shape=(32,32,3))
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(3,3))(x)
  x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
  x = tf.keras.layers.Flatten()(x)
  for i in range(10):
    x = tf.keras.layers.Dense(144)(x)
    x = tf.keras.layers.ELU()(x)
  x = tf.keras.layers.Dense(10, activation="softmax")(x)

  model = tf.keras.models.Model(input=inp, output=x)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metric='accuracy')

  model.summary()
  est = tf.keras.estimator.model_to_estimator(keras_model=model)

  #classifier = tf.estimator.Estimator(
  #  model_fn=model_fn,
  #  params={
  #    'feature_columns'
  #    'n_class'}

  return est 


if __name__=="__main__":
  input_fn = data_pipeline.input_fn_gen()

  sess = tf.Session()
  image, label = input_fn({"batch_size":64})

  est = build_model() 

