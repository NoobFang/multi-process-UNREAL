from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import namedtuple
import scipy.signal
import threading
from 
from experience import Experience, ExperienceFrame
from constants import *

def choose_action(pi):
  return np.random.choice(range(len(pi)), p=pi)

class RunnerThread(threading.Thread):
  """
  provide the interface which used to interact with environment
  """
  def __init__(self, env, policy, num_local_steps, visualise):
    threading.Thread.__init__(self)
    self.experience = Experience(EXPERIENCE_HISTORY_SIZE)
    self.queue = queue.Queue(5)
    self.num_local_steps = num_local_steps
    self.env = env
    self.last_features = None
    self.policy = policy
    self.daemon = True
    self.sess = None
    self.summary_writer = None
    self.visualise = visualise

  def start_runner(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer
    self.start()
  
  def fill_experience(self, sess):
    while not self.experience.is_full():
      last_state = self.env.last_state
      last_action = self.env.last_action
      last_reward = self.env.last_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.env.get_action_size(),
                                                                    last_reward)
      pi_, _ = self.policy.run_base_policy_and_value(sess,
                                                    self.env.last_state,
                                                    last_action_reward)
      action = choose_action(pi_)
      state, reward, terminal, pixel_change = self.env.process(action)
      frame = ExperienceFrame(last_state, reward, action, terminal, pixel_change,
                              last_action, last_reward)
      self.experience.add_frame(frame)

      if terminal:
        self.env.reset()
    self.env.reset()
    print('Replay buffer has been filled')

  def run(self):
    with self.sess.as_default():
      self.fill_experience(self.sess)
      data_provider = env_runner(self.env, self.policy, self.num_local_steps, 
                                 self.summary_writer, self.visualise)
      while True:
        # add experience frame into exp pool
        rollout = next(data_provider)
        self.queue.put(rollout, timeout=600.0)
        self.experience.add_rollout(rollout)

  def env_runner(self, env, policy, num_local_steps, summary_writer, visualise):
    """
    The logic of the thread runner.
    """
    last_state = env.reset()

    while True:
      terminal_end = False
      rollout = []
      for _ in range(num_local_steps):
        last_state = env.last_state
        last_action = env.last_action
        last_reward = env.last_reward
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                      env.get_action_size(),
                                                                      last_reward)
        fetched = policy.run_base_policy_and_value(self.sess, last_state, last_action_reward)
        pi_, value_ = fetched[0], fetched[1]
        action = choose_action(pi_)
        
        state, reward, terminal, pixel_change = env.process(action)
        # TODO: render the environment
        # TODO: collect experience and use it to fill replay pool
        if not terminal:
          frame = ExperienceFrame(last_state, reward, action, terminal, pixel_change,
                                  last_action, last_reward)
          rollout.append(frame)
        else:
          env.reset()
          break
      yield rollout

  