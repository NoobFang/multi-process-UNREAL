from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import UnrealNetwork

class UNREAL(object):
  def __init__(self, env, task, visualise):
    self.env = env
    self.task = task
    worker_device = '/job:worker/task:{}'.format(task)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
      with tf.variable_scope('global'):
        self.network = UnrealNetwork(env.observation_space.shape, 
                                    env.action_space.n)
        self.global_step = tf.get_variable('global_step', [], tf.int32, 
                                          initializer=tf.constant(0, dtype=tf.int32),
                                          trainable=False)

    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.local_network = pi = UnrealNetwork(env.observation_space.shape, 
                                           env.action_space.n)
        pi.global_step = self.global_step
      
      # TODO: define the input and loss of UNREAL model