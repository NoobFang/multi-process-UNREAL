from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.model import UnrealModel

class UNREAL(object):
  def __init__(self, env, task, visualise):
    