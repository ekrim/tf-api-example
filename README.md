I like TensorFlow's Dataset API because it is super optimized for asynchronous data transfer to the GPU from files on disk. I can never get utilization as high with other data readers. The Estimator API is a little hard to work with, but it is needed to work on a TPU.

In this repo, I train a model on CIFAR-10 using TensorFlow's current high-level APIs. Use this as a basic reference for converting binary data to TFRecords, using multithreaded dataset readers, building estimators for train/eval/predict, and creating logging hooks.

To train, simply run:

```
python main.py
```

The dataset is downloaded and converted into tfrecords automatically if they are not already there.
