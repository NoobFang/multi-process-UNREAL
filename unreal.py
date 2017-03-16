from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import UnrealNetwork
from constants import *

class UNREAL(object):
  def __init__(self, env, task, visualise):
    self.env = env
    self.task = task
    # TODO:get the observation shape and action number
    self.ob_shape
    self.action_n
    # define the network stored in ps which is used to sync
    worker_device = '/job:worker/task:{}'.format(task)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
      with tf.variable_scope('global'):
        self.network = UnrealNetwork(env.observation_space.shape, 
                                    env.action_space.n)
        self.global_step = tf.get_variable('global_step', [], tf.int32, 
                                          initializer=tf.constant(0, dtype=tf.int32),
                                          trainable=False)
    # define the local network which is used to calculate the gradient
    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.local_network = pi = UnrealNetwork(env.observation_space.shape, 
                                           env.action_space.n)
        pi.global_step = self.global_step
      
      self.a = tf.placeholder(dtype=tf.int32, [None, self.action_n]) # The taken action of a3c
      self.adv = tf.placeholder(dtype=tf.float32, [None]) # Advantage of a3c
      self.r = tf.placeholder(dtype=tf.float32, [None]) # Reward of a3c
      self.pc_a = tf.placeholder(dtype=tf.int32, [None, self.action_n]) # The sampled action for pixel control task
      self.pc_r = tf.placeholder(dtype=tf.float32, [None, 20, 20]) # TD target of pixel control task
      self.rp_c = tf.placeholder(dtype=tf.int32, [1,3]) # one-hot target of reward prediction task
      self.vr_r = tf.placeholder(dtype=tf.float32, [None])
      
      # define the loss of base-a3c agent
      def _base_loss(self):
        log_pi = tf.log(tf.clip_by_value(pi.base_pi, 1e-12, 1.0))
        entropy = - tf.reduce_sum(pi.base_pi*log_pi, axis=1)
        policy_loss = - tf.reduce_sum(tf.reduce_sum(log_pi*self.a, axis=1)*self.adv)
        value_loss = 0.5 * tf.nn.l2_loss(pi.base_v - self.r)
        return policy_loss + value_loss - entropy * ENTROPY_BETA

      # define the loss of pixel control task
      def _pc_loss(self):
        reshaped_a = tf.reshape(self.pc_a, [-1, 1, 1, self.action_n])
        qa = tf.reduce_sum(pi.pc_q * reshaped_a, axis=3)
        pc_loss = PIXEL_CHANGE_LAMBDA * tf.nn.l2_loss(self.pc_r - qa)
        return pc_loss
      
      #define the loss of reward prediction task
      def _rp_loss(self):
        rp_c = tf.clip_by_value(pi.rp_c, 1e-12, 1.0)
        rp_loss = - tf.reduce_sum(self.rp_c * tf.log(rp_c))
        return rp_loss

      # define the loss of value function replay task
      def _vr_loss(self):
        vr_loss = tf.nn.l2_loss(self.vr_r - pi.vr_v)
        return vr_loss

      # TODO: using multiple runner threads to run policy interact with
      # environment to get rollouts and fill the exp buffer
      self.runner = RunnerThread(env, pi, 20, visualise) # 20 is the number of time-steps considered in lstm network
      