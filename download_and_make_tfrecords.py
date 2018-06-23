import struct
import os
import tarfile
import urllib
import shutil
from glob import glob
import numpy as np
import tensorflow as tf


def maybe_download_cifar():
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
  base_dir = 'data'
  fn_gz = 'cifar-10-binary.tar.gz'
  fn_x = 'data_batch_1.bin'
  fn_tf = 'data_batch_1.tfrecords'
  dir_name = 'cifar-10-batches-bin'

  add_path = lambda x: os.path.join(base_dir, x)
  exists = lambda x: os.path.exists(add_path(x))

  if not os.path.exists(base_dir):
    os.mkdir(base_dir)

  if not exists(fn_gz) and not exists(fn_x) and not exists(fn_tf):
    print('downloading the tar.gz cifar file')
    urllib.request.urlretrieve(url, add_path(fn_gz))  

  if not exists(fn_x) and not exists(fn_tf):
    print('extracting tar.gz and cleaning up')
    with tarfile.open(add_path(fn_gz)) as f:
      f.extractall()

    for fn in glob(os.path.join(dir_name, '*')):
      shutil.move(fn, base_dir)

    shutil.rmtree(dir_name)
    os.remove(add_path(fn_gz))

  if not exists(fn_tf):
    print('converting to tfrecords')
    to_tfrecords(base_dir)
   

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

 
def to_tfrecords(dir_name): 
  input_files = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name) if fn.endswith(".bin")]
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


if __name__ == '__main__':
  maybe_download_cifar()
