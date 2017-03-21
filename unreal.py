from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model import UnrealModel
from experience import Experience, ExperienceFrame
from environment.environment import Environment
from env_runner import RunnerThread
from constants import *

# sample learning rate or weight for UNREAL model for log-uniform distribution
def log_uniform(lo, hi):
  r = np.random.uniform(low=0.0, high=1.0)
  out = np.exp(np.log(lo)*(1-r) + np.log(hi)*r)
  return out

# TODO: convert the data for a3c, pc, vr, rp correctly
def discount(x, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

Batch = namedtuple('Batch', ['s', 'a', 'adv', 'r', 'terminal', 'features'])

def process_rollout(rollout, gamma, lambda_=1.0):
  '''
  given a rollout, compute its returns and advantage
  '''
  batch_s = np.asarray(rollout.states)
  batch_a = np.asarray(rollout.actions)
  rewards = np.asarray(rollout.rewards)
  v_predt = np.asarray(rollout.values + [rollout.r])
  rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])

  batch_r = discount(rewards_plus_v, gamma)[:-1]
  delta_t = rewards + gamma*v_predt[1:] - v_predt[:-1]
  batch_adv = discount(delta_t, gamma*lambda_)
  features = rollout.features[0]

  return Batch(batch_s, batch_a, batch_adv, batch_r, rollout.terminal, features)

class UNREAL(object):
  def __init__(self, env, task, visualise):
    self.env = env
    self.task = task
    self.ob_shape = [HEIGHT, WIDTH, CHANNEL]
    self.action_n = Environment.get_action_size()
    self.experience = Experience(EXPERIENCE_HISTORY_SIZE) # exp replay pool
    # define the network stored in ps which is used to sync
    worker_device = '/job:worker/task:{}'.format(task)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
      with tf.variable_scope('global'):
        self.network = UnrealModel(self.action_n)
        self.global_step = tf.get_variable('global_step', [], tf.int32, 
                                          initializer=tf.constant(0, dtype=tf.int32),
                                          trainable=False)
    # define the local network which is used to calculate the gradient
    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.local_network = pi = UnrealModel(self.action_n)
        pi.global_step = self.global_step

      # define the runner handle the interaction with the environment
      self.runner = RunnerThread(env, pi, 20, visualise) # 20 is the number of time-steps considered in lstm network
      
      self.loss = pi.prepare_loss()
      
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


def process_a3c_data(rollout):
  """
  recieve a rollout from env_runner and convert it into batch data for a3c
  """
  states = []
  last_action_rewards = []
  actions = []
  rewards = []
  values = []

  for frame in rollout:
    states.append(frame.state)
    last_action = frame.last_action
    last_reward = frame.last_reward
    last_action_rewards.append(ExperienceFrame.concat_action_and_reward(last_action,
                                                                        self.env.get_action_size(),
                                                                        last_reward))
    actions.append(frame.action)
    rewards.append(frame.reward)
    