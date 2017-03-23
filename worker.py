from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import logging
import argparse
import sys, signal, time, os
from environment.environment import Environment
from unreal import UNREAL
from constants import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(args, server):
  # create an environment chosed by global ENV_TYPE
  env = Environment.create_environment()
  trainer = UNREAL(env, args.task, args.visualise)

  variables_to_save = [v for v in tf.global_variables() if not v.name.startswith('local')]
  init_op = tf.variables_initializer(variables_to_save)
  init_all_op = tf.global_variables_initializer()
  saver = tf.train.Saver(variables_to_save)

  def init_fn(sess):
    logger.info('Initializing all parameters...')
    sess.run(init_all_op)

  config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:{}'.format(args.task)])
  logdir = os.path.join(args.log_dir, 'train')
  summary_writer = tf.summary.FileWriter(logdir + '_%d' % args.task)
  logger.info('Event directory: %s_%s', logdir, args.task)
  
  sv = tf.train.Supervisor(is_chief = (args.task == 0),
                           logdir = logdir,
                           saver = saver,
                           summary_op = None,
                           init_op = init_op,
                           init_fn = init_fn,
                           summary_writer=summary_writer,
                           ready_op = tf.report_uninitialized_variables(variables_to_save),
                           global_step = trainer.global_step,
                           save_model_secs=600,
                           save_summaries_secs=120)
  
  num_global_steps = MAX_TRAIN_STEP

  logger.info(
    'Starting session...\n'+'If this hangs, we are mostly likely waiting to' +
    'connect to the parameter server.'
  )
  with sv.managed_session(server.target, config=config) as sess:
    sess.as_default()
    trainer.start(sess, summary_writer)
    global_step = sess.run(trainer.global_step)
    logger.info('Starting training at step=%d'%global_step)
    while not sv.should_stop() and global_step < num_global_steps:
      trainer.process(sess)
      global_step = sess.run(trainer.global_step)
    sv.stop()
    logger.info('reached %s steps. worker stopped.' % global_step)

# create a cluster specification
def cluster_spec(num_workers, num_ps):
  cluster = {}
  host = '127.0.0.1'
  port = 12222

  all_ps = []
  for _ in range(num_ps):
    all_ps.append('{}:{}'.format(host, port))
    port += 1
  cluster['ps'] = all_ps

  all_workers= []
  for _ in range(num_workers):
    all_workers.append('{}:{}'.format(host, port))
    port += 1
  cluster['worker'] = all_workers

  return cluster

def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--verbose', default=0, action='count',
                      dest='verbosity', help='set verbosity')
  parser.add_argument('--task', default=0, type=int, help='task index')
  parser.add_argument('--job-name', default='worker', help='worker or ps')
  parser.add_argument('--num-workers', default=1, type=int,
                      help='number of workers')
  parser.add_argument('--log-dir', default='/tmp/unreal',
                      help='log directory path')
  parser.add_argument('--env-id', default='seekavoid_arena_01',
                      help='environment name')
  parser.add_argument('--visualise', action='store_true',
                      help='whether visualise the environment')

  args = parser.parse_args()
  # using 1 ps process by default
  cluster = cluster_spec(args.num_workers, 1)
  
  def shutdown(signal, frame):
    logger.warn('Received signal %s: exiting', signal)
    sys.exit(128+signal)
  signal.signal(signal.SIGHUP, shutdown)
  signal.signal(signal.SIGINT, shutdown)
  signal.signal(signal.SIGTERM, shutdown)

  if args.job_name == 'ps':
    config = tf.ConfigProto(device_filters=['/job:ps'])
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name='ps', task_index=args.task, config=config)
    server.join()
  else: # worker jobs
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name='worker', task_index=args.task, config=config)
    run(args, server)

if __name__ == '__main__':
  tf.app.run()
    

  