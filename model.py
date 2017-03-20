# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
import numpy as np

from constants import *

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

class UnrealNetwork(object):
  def __init__(self, ob_shape, ac_shape):
    self.ob_shape = ob_shape
    self.action_n = ac_shape
    self.create_a3c()

  def create_a3c(self):
    self.image_input = tf.placeholder(dtype=tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    self.last_action_reward_input = tf.placeholder(dtype=tf.float32, [None, self.action_n+1])
    self.conv


  def create_pc(self)

  def create_rp(self)

  def create_vr(self)
  
