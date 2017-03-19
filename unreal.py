from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf\
import six.moves.queue as queue
import scipy.signal
from model import UnrealNetwork
from environment.environment import Environment
from constants import *

# sample learning rate or weight for UNREAL model for log-uniform distribution
def log_uniform(lo, hi):
  r = np.random.uniform(low=0.0, high=1.0)
  out = np.exp(np.log(lo)*(1-r) + np.log(hi)*r)
  return out

# TODO: define the environment runner and rollout and experience buffer
def discount(x, gamma):
  return scipy.signal.lfilter()

class UNREAL(object):
  def __init__(self, env, task, visualise):
    self.env = env
    self.task = task
    self.ob_shape = [HEIGHT, WIDTH, CHANNEL]
    self.action_n = Environment.get_action_size()
    # define the network stored in ps which is used to sync
    worker_device = '/job:worker/task:{}'.format(task)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
      with tf.variable_scope('global'):
        self.network = UnrealNetwork(self.ob_shape, self.action_n)
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
      
      # define the entropy of policy action output
      entropy = - tf.reduce_sum(pi.base_pi*log_pi, axis=1)
      
      # define the loss of base-a3c agent
      log_pi = tf.log(tf.clip_by_value(pi.base_pi, 1e-12, 1.0))
      policy_loss = - tf.reduce_sum(tf.reduce_sum(log_pi*self.a, axis=1)*self.adv)
      value_loss = 0.5 * tf.nn.l2_loss(pi.base_v - self.r)
      entropy_weight = log_uniform(EN_LOW, EN_HIGH)
      base_loss = policy_loss + value_loss - entropy*entropy_weight
      
      # define the loss of pixel control task
      pc_loss = 0.0
      if USE_PIXEL_CHANGE:
        reshaped_a = tf.reshape(self.pc_a, [-1, 1, 1, self.action_n])
        qa = tf.reduce_sum(pi.pc_q * reshaped_a, axis=3)
        pc_weight = log_uniform(PC_LOW, PC_HIGH)
        pc_loss = pc_weight * tf.nn.l2_loss(self.pc_r - qa)
      
      #define the loss of reward prediction task
      rp_loss = 0.0
      if USE_REWARD_PREDICTION:
        rp_c = tf.clip_by_value(pi.rp_c, 1e-12, 1.0)
        rp_loss = - tf.reduce_sum(self.rp_c * tf.log(rp_c))

      # define the loss of value function replay task
      vr_loss = 0.0
      if USE_VALUE_REPLAY:
        vr_loss = tf.nn.l2_loss(self.vr_r - pi.vr_v)

      # define the runner handle the interaction with the environment
      self.runner = RunnerThread(env, pi, 20, visualise) # 20 is the number of time-steps considered in lstm network
      
      # get the gradients by local network computation
      self.loss = base_loss + pc_loss + rp_loss + vr_loss
      grads = tf.gradients(self.loss, pi.var_list)

      # add summaries for losses and norms
      batch_size = tf.to_float(tf.shape(pi.x)[0])
      tf.summary.scalar('model/a3c_loss', base_loss / batch_size)
      tf.summary.scalar('model/pc_loss', pc_loss / batch_size)
      tf.summary.scalar('model/rp_loss', rp_loss / batch_size)
      tf.summary.scalar('model/vr_loss', vr_loss / batch_size)
      tf.summary.scalar('model/grad_global_norm', tf.global_norm(grads))
      tf.summary.scalar('model/var_global_norm', tf.global_norm(pi.var_list))
      tf.summary.scalar('model/entropy', entropy / batch_size)
      tf.summary.image('model/state', pi.x)
      self.summary_op = tf.summary.merge_all()
      
      # clip the gradients to avoid gradient explosion
      grads, _ = tf.clip_by_global_norm(grads, GRAD_NORM_CLIP)

      self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])
      grads_and_vars = list(zip(grads, self.network.var_list))
      inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
      lr = log_uniform(LR_LOW, LR_HIGH)
      opt = tf.train.RMSPropOptimizer(learning_rate=lr,
                                      decay=RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=RMSP_EPSILON)
      self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
      self.summary_writer = None
      self.local_step = 0
    
  def start(self, sess, summary_writer):
    self.runner.start_runner(sess, summary_writer)
    self.summary_writer = summary_writer

  def pull_batch_from_queue(self):
    rollout = self.runner.queue.get(timeout=600.0)
    while not rollout.terminal:
      try:
        rollout.extend(self.runner.queue.get_nowait())
      except queue.Empty:
        break
    return rollout

  def process(self, sess):
    sess.run(self.sync) # sync the local network with the global network
    rollout = self.pull_batch_from_queue()
    batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

    should_summarize = self.task==0 and self.local_step%11==0

    if should_summarize:
      fetches = [self.summary_op, self.train_op, self.global_step]
    else:
      fetches = [self.train_op, self.global_step]
    
    feed_dict = {
      self.local_network.x: batch.state,
      self.a: batch.a,
      self.adv: batch.adv,
      self.r: batch.r,
      self.local_network.state_in[0]: batch.features[0],
      self.local_network.state_in[1]: batch.features[1]
    }

    fetched = sess.run(fetches, feed_dict=feed_dict)

    if should_summarize:
      self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
      self.summary_writer.flush()
    self.local_step += 1