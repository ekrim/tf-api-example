import struct
import os
import pickle
import numpy as np
import tensorflow as tf


def cifar_reader(f):
  byte_str = f.read(3073) 
  if byte_str==b"":
    return None, None
  else:
    record_tuple = struct.unpack("3073B", byte_str)
    label = record_tuple[0]
    img = np.array(record_tuple[1:]).astype(np.uint8)
    #img = np.concatenate(np.split(img.reshape((32*3,32))[:,:,None], 3, axis=0), axis=2)
    return img, label


if __name__=="__main__":
  
  input_files = [fn for fn in os.listdir() if fn.endswith(".bin")]
  output_files = [os.path.splitext(fn)[0]+".tfrecords" for fn in input_files]

  for input_file, output_file in zip(input_files, output_files):
    with open(input_file, "rb") as f:
      with tf.python_io.TFRecordWriter(output_file) as record_writer:
        while True:
          img, label = cifar_reader(f)

          if img is None:
            break
          
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
              }))

          record_writer.write(example.SerializeToString())
