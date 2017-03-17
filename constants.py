# -*- coding: utf-8 -*-
LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.001 # epsilon parameter for RMSProp
CHECKPOINT_DIR = '/tmp/unreal_checkpoints'
LOG_FILE = '/tmp/unreal_log/unreal_log'
LR_LOW = 1e-4    # log_uniform low limit for learning rate
LR_HIGH = 5e-3   # log_uniform high limit for learning rate
EN_LOW = 5e-4    # log_uniform low limit for entropy cost
EN_HIGH = 0.01   # log_uniform high limit for entropy cost
PC_LOW = 0.01    # log_uniform low limit for pixel control loss weight
PC_HIGH = 0.1    # log_uniform high limit for pixel control loss weight
PARALLEL_SIZE = 8 # parallel thread size
MAX_TRAIN_STEP = 1e8

ENV_TYPE = 'lab' # 'lab' or 'gym' or 'maze'
ENV_NAME = 'seekavoid_arena_01'
#ENV_NAME = 'stairway_to_melon'
#ENV_NAME = 'nav_maze_static_01'

#ENV_TYPE = 'gym'
#ENV_NAME = 'Breakout-v0'

GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.001 # entropy regurarlization constant, sample from a log_uniform between 0.0001 and 0.005
PIXEL_CHANGE_LAMBDA = 0.05 # 0.01 ~ 0.1 for Lab, 0.0001 ~ 0.01 for Gym
EXPERIENCE_HISTORY_SIZE = 2000 # Experience replay buffer size

USE_PIXEL_CHANGE      = True
USE_VALUE_REPLAY      = True
USE_REWARD_PREDICTION = True

MAX_TIME_STEP = 10 * 10**7
SAVE_INTERVAL_STEP = 100 * 1000

GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True

HEIGHT = 84
WIDTH = 84
CHANNEL = 3
