import numpy as np
import tensorflow as tf

def write_tfrecord(filename, img):
  writer = tf.python_io.TFRecordWriter(filename)
  feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())]))}
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example.SerializeToString())
  writer.close()

def read_tfrecord(filename):
  feature = {'image': tf.FixedLenFeature([], tf.string)}
  filename_queue = tf.train.string_input_producer([filename], num_epochs=1)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features=feature)
  # Convert the image data from string back to the numbers
  image = tf.decode_raw(features['image'], tf.uint8)
  
  # Cast label data into int32
  # Reshape image data into the original shape
  image = tf.reshape(image, [2, 2, 3])
  
  # Any preprocessing here ...
  
  # Creates batches by randomly shuffling tensors
  #images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
  return image


if __name__=='__main__':
  x = (np.zeros((2,2,3)) + np.arange(3)[None,None,:]).astype(np.uint8)
  fn = "mytest.tfrecords"
  
  write_tfrecord(fn, x)
  output = read_tfrecord(fn)
  print(output)

  with tf.Session().as_default():
    print(output.eval())
  #with tf.Session() as sess:
  #  you = sess.run(output)

  #print(you)
