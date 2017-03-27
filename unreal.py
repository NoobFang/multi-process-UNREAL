from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model import UnrealModel
from experience import Experience, ExperienceFrame
from environment.environment import Environment
from constants import *

# sample learning rate or weight for UNREAL model for log-uniform distribution
def log_uniform(lo, hi):
  r = np.random.uniform(low=0.0, high=1.0)
  out = np.exp(np.log(lo)*(1-r) + np.log(hi)*r)
  return out

def choose_action(pi):
  return np.random.choice(range(len(pi)), p=pi)

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
        self.experience = Experience(EXPERIENCE_HISTORY_SIZE) # exp replay pool
        self.network = UnrealModel(self.action_n, self.env, self.experience)
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, 
                                          initializer=tf.constant(0, dtype=tf.int32),
                                          trainable=False)
    # define the local network which is used to calculate the gradient
    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.local_network = net = UnrealModel(self.action_n, self.env, self.experience)
        net.global_step = self.global_step
    
      # add summaries for losses and norms
      self.batch_size = tf.to_float(tf.shape(net.base_input)[0])
      base_loss = self.local_network.base_loss
      pc_loss = self.local_network.pc_loss
      rp_loss = self.local_network.rp_loss
      vr_loss = self.local_network.vr_loss
      entropy = tf.reduce_sum(self.local_network.entropy)
      self.loss = base_loss + pc_loss + rp_loss + vr_loss
      grads = tf.gradients(self.loss, net.var_list)
      tf.summary.scalar('model/a3c_loss', base_loss / self.batch_size)
      tf.summary.scalar('model/pc_loss', pc_loss / self.batch_size)
      tf.summary.scalar('model/rp_loss', rp_loss / self.batch_size)
      tf.summary.scalar('model/vr_loss', vr_loss / self.batch_size)
      tf.summary.scalar('model/grad_global_norm', tf.global_norm(grads))
      tf.summary.scalar('model/var_global_norm', tf.global_norm(net.var_list))
      tf.summary.scalar('model/entropy', entropy / self.batch_size)
      tf.summary.image('model/state', net.base_input)
      self.summary_op = tf.summary.merge_all()
      
      # clip the gradients to avoid gradient explosion
      grads, _ = tf.clip_by_global_norm(grads, GRAD_NORM_CLIP)

      self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(net.var_list, self.network.var_list)])
      grads_and_vars = list(zip(grads, self.network.var_list))
      inc_step = self.global_step.assign_add(tf.to_int32(self.batch_size))
      lr = log_uniform(LR_LOW, LR_HIGH)
      opt = tf.train.RMSPropOptimizer(learning_rate=lr,
                                      decay=RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=RMSP_EPSILON)
      self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
      self.summary_writer = None
      self.local_step = 0
    
  def start(self, sess, summary_writer):
    self.summary_writer = summary_writer
  
  def fill_experience(self, sess):
    """
    Fill experience buffer until buffer is full.
    """
    prev_state = self.env.last_state
    last_action = self.env.last_action
    last_reward = self.env.last_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                  self.action_n,
                                                                  last_reward)
    
    pi_, _ = self.local_network.run_base_policy_and_value(sess,
                                                          prev_state,
                                                          last_action_reward)
    action = choose_action(pi_)
    
    new_state, reward, terminal, pixel_change = self.env.process(action)
    
    frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                            last_action, last_reward)
    self.experience.add_frame(frame)
    
    if terminal:
      self.env.reset()
    if self.experience.is_full():
      self.env.reset()
      print("Replay buffer filled")

  def process(self, sess):
    sess.run(self.sync) # sync the local network with the global network
    if not self.experience.is_full():
      self.fill_experience(sess)
      return 0
    
    batch_data = self.local_network.get_batch_data(sess)

    feed_dict = {
      self.local_network.base_input: batch_data[0],
      self.local_network.base_last_action_reward_input: batch_data[1],
      self.local_network.base_a: batch_data[2],
      self.local_network.base_adv: batch_data[3],
      self.local_network.base_r: batch_data[4],
      self.local_network.base_initial_lstm_state: batch_data[5],
      self.local_network.pc_input: batch_data[6],
      self.local_network.pc_last_action_reward_input: batch_data[7],
      self.local_network.pc_a: batch_data[8],
      self.local_network.pc_r: batch_data[9],
      self.local_network.vr_input: batch_data[10],
      self.local_network.vr_last_action_reward_input: batch_data[11],
      self.local_network.vr_r: batch_data[12],
      self.local_network.rp_input: batch_data[13],
      self.local_network.rp_r: batch_data[14]
    }

    should_summarize = self.task==0 and self.local_step%11==0

    if should_summarize:
      fetches = [self.summary_op, self.train_op, self.global_step]
    else:
      fetches = [self.train_op, self.global_step]

    fetched = sess.run(fetches, feed_dict=feed_dict)

    if should_summarize:
      self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
      self.summary_writer.flush()
    self.local_step += 1
