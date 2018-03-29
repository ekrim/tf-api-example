from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def model_fn(features, labels, mode, params):
  
