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
    self.episode_reward = 0
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
        self.local_network = net = UnrealModel(self.action_n)
        pi.global_step = self.global_step

      # define the runner handle the interaction with the environment
      self.runner = RunnerThread(env, net, 20, visualise) # 20 is the number of time-steps considered in lstm network
      
      # add summaries for losses and norms
      self.batch_size = tf.to_float(tf.shape(net.base_input)[0])
      self.loss = self.prepare_loss()
      grads = tf.gradients(self.loss, net.var_list)
      tf.summary.scalar('model/grad_global_norm', tf.global_norm(grads))
      tf.summary.scalar('model/var_global_norm', tf.global_norm(net.var_list))
      tf.summary.scalar('model/entropy', self.entropy / self.batch_size)
      tf.summary.image('model/state', net.base_input)
      self.summary_op = tf.summary.merge_all()
      
      # clip the gradients to avoid gradient explosion
      grads, _ = tf.clip_by_global_norm(grads, GRAD_NORM_CLIP)

      self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(net.var_list, self.network.var_list)])
      grads_and_vars = list(zip(grads, self.network.var_list))
      inc_step = self.global_step.assign_add(self.batch_size)
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
    action = self.choose_action(pi_)
    
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
    
    # [Base]
    batch_si, batch_last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state = \
      self._process_a3c(sess)
    feed_dict = {
      self.local_network.base_input: batch_si,
      self.local_network.base_last_action_reward_input: batch_last_action_rewards,
      self.local_network.base_a: batch_a,
      self.local_network.base_adv: batch_adv,
      self.local_network.base_r: batch_R,
      self.local_network.base_initial_lstm_state: start_lstm_state
    }

    # [Pixel change]
    if USE_PIXEL_CHANGE:
      batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc(sess)

      pc_feed_dict = {
        self.local_network.pc_input: batch_pc_si,
        self.local_network.pc_last_action_reward_input: batch_pc_last_action_reward,
        self.local_network.pc_a: batch_pc_a,
        self.local_network.pc_r: batch_pc_R
      }
      feed_dict.update(pc_feed_dict)

    # [Value replay]
    if USE_VALUE_REPLAY:
      batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr(sess)
      
      vr_feed_dict = {
        self.local_network.vr_input: batch_vr_si,
        self.local_network.vr_last_action_reward_input : batch_vr_last_action_reward,
        self.local_network.vr_r: batch_vr_R
      }
      feed_dict.update(vr_feed_dict)

    # [Reward prediction]
    if USE_REWARD_PREDICTION:
      batch_rp_si, batch_rp_c = self._process_rp()
      rp_feed_dict = {
        self.local_network.rp_input: batch_rp_si,
        self.local_network.rp_c_target: batch_rp_c
      }
      feed_dict.update(rp_feed_dict)

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

  def _process_a3c(self, sess):
    states = []
    last_action_rewards = []
    actions = []
    rewards = []
    values = []
    terminal_end = False

    for _ in range(LOCAL_T_MAX):
      last_action = self.env.last_action
      last_reward = self.env.last_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.action_n,
                                                                    last_reward)
      pi_, value_ = self.local_network.run_base_policy_and_value(sess,
                                                                self.env.last_state,
                                                                last_action_reward)
      action = self.choose_action(pi_)
      states.append(self.env.last_state)
      last_action_rewards.append(last_action_reward)
      actions.append(action)
      values.append(value_)

      prev_state = self.env.last_state

      # Process game
      new_state, reward, terminal, pixel_change = self.env.process(action)
      frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                              last_action, last_reward)

      # Store to experience
      self.experience.add_frame(frame)

      self.episode_reward += reward

      rewards.append(reward)

      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))
          
        self.episode_reward = 0
        self.env.reset()
        self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_base_value(sess, new_state, frame.get_last_action_reward(self.action_n))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_adv = []
    batch_R = []

    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      adv = R - Vi
      a = np.zeros([self.action_n])
      a[ai] = 1.0

      batch_si.append(si)
      batch_a.append(a)
      batch_adv.append(adv)
      batch_R.append(R)

    batch_si.reverse()
    batch_a.reverse()
    batch_adv.reverse()
    batch_R.reverse()
    
    return batch_si, last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state

  def _process_pc(self, sess):
    # [pixel change]
    # Sample 20+1 frame (+1 for last next state)
    pc_experience_frames = self.experience.sample_sequence(LOCAL_T_MAX+1)
    # Revese sequence to calculate from the last
    pc_experience_frames.reverse()

    batch_pc_si = []
    batch_pc_a = []
    batch_pc_R = []
    batch_pc_last_action_reward = []
    
    pc_R = np.zeros([20,20], dtype=np.float32)
    if not pc_experience_frames[0].terminal:
      pc_R = self.local_network.run_pc_q_max(sess,
                                             pc_experience_frames[0].state,
                                             pc_experience_frames[0].get_last_action_reward(self.action_size))


    for frame in pc_experience_frames[1:]:
      pc_R = frame.pixel_change + GAMMA * pc_R
      a = np.zeros([self.action_n])
      a[frame.action] = 1.0
      last_action_reward = frame.get_last_action_reward(self.action_n)
      
      batch_pc_si.append(frame.state)
      batch_pc_a.append(a)
      batch_pc_R.append(pc_R)
      batch_pc_last_action_reward.append(last_action_reward)

    batch_pc_si.reverse()
    batch_pc_a.reverse()
    batch_pc_R.reverse()
    batch_pc_last_action_reward.reverse()
    
    return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R

  def _process_vr(self, sess):
    # [Value replay]
    # Sample 20+1 frame (+1 for last next state)
    vr_experience_frames = self.experience.sample_sequence(LOCAL_T_MAX+1)
    # Revese sequence to calculate from the last
    vr_experience_frames.reverse()

    batch_vr_si = []
    batch_vr_R = []
    batch_vr_last_action_reward = []

    vr_R = 0.0
    if not vr_experience_frames[0].terminal:
      vr_R = self.local_network.run_vr_value(sess,
                                             vr_experience_frames[0].state,
                                             vr_experience_frames[0].get_last_action_reward(self.action_size))
    
    # t_max times loop
    for frame in vr_experience_frames[1:]:
      vr_R = frame.reward + GAMMA * vr_R
      batch_vr_si.append(frame.state)
      batch_vr_R.append(vr_R)
      last_action_reward = frame.get_last_action_reward(self.action_n)
      batch_vr_last_action_reward.append(last_action_reward)

    batch_vr_si.reverse()
    batch_vr_R.reverse()
    batch_vr_last_action_reward.reverse()

    return batch_vr_si, batch_vr_last_action_reward, batch_vr_R

  def _process_rp(self):
    # [Reward prediction]
    rp_experience_frames = self.experience.sample_rp_sequence()
    # 4 frames

    batch_rp_si = []
    batch_rp_c = []
    
    for i in range(3):
      batch_rp_si.append(rp_experience_frames[i].state)

    # one hot vector for target reward
    r = rp_experience_frames[3].reward
    rp_c = [0.0, 0.0, 0.0]
    if r == 0:
      rp_c[0] = 1.0 # zero
    elif r > 0:
      rp_c[1] = 1.0 # positive
    else:
      rp_c[2] = 1.0 # negative
    batch_rp_c.append(rp_c)
    return batch_rp_si, batch_rp_c


  def prepare_loss(self):
    loss = self.base_loss()
    tf.summary.scalar('model/a3c_loss', self.base_loss() / self.batch_size)
    if USE_PIXEL_CHANGE:
      loss += self.pc_loss()
      tf.summary.scalar('model/pc_loss', self.pc_loss() / self.batch_size)
    if USE_VALUE_REPLAY:
      loss += self.vr_loss()
      tf.summary.scalar('model/rp_loss', self.rp_loss() / self.batch_size)
    if USE_REWARD_PREDICTION:
      loss += self.rp_loss()
      tf.summary.scalar('model/vr_loss', self.vr_loss() / self.batch_size)
    return loss

  def base_loss(self):
    net = self.local_network
    self.base_a = tf.placeholder(dtype=tf.float32, [None, self.action_n])
    self.base_adv = tf.placeholder(dtype=tf.float32, [None])
    self.base_r = tf.placeholder(dtype=tf.float32, [None])

    log_pi = tf.log(tf.clip_by_value(net.base_pi, 1e-20, 1.0))
    self.entropy = - tf.reduce_sum(net.base_pi*log_pi, axis=1)
    policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.base_a),axis=1)*
                                  self.base_adv + self.entropy * ENTROPY_BETA)
    value_loss = 0.5 * tf.nn.l2_loss(self.base_r - net.base_v)
    base_loss = policy_loss + value_loss
    return base_loss

  def pc_loss(self):
    net = self.local_network
    self.pc_a = tf.placeholder(dtype=tf.float32, [None, self.action_n])
    self.pc_r = tf.placeholder(dtype=tf.float32, [None, 20, 20])
    pc_a_reshape = tf.reshape(self.pc_a, [-1, 1, 1, self.action_n])
    pc_qa_ = tf.multiply(net.pc_q, pc_a_reshape)
    pc_qa = tf.reduce_sum(pc_qa_, axis=3)
    pc_loss = tf.nn.l2_loss(self.pc_r - pc_qa) * PIXEL_CHANGE_LAMBDA
    return pc_loss
  
  def vr_loss(self):
    net = self.local_network
    self.vr_r = tf.placeholder(dtype=tf.float32, [None])
    vr_loss = tf.nn.l2_loss(self.vr_r - net.vr_v)
    return vr_loss

  def rp_loss(self):
    net = self.local_network
    self.rp_r = tf.placeholder(dtype=tf.float32, [1,3])
    rp_c = tf.clip_by_value(net.rp_c, 1e-20, 1.0)
    rp_loss = - tf.reduce_sum(self.rp_r * tf.log(rp_c))
    return rp_loss


    