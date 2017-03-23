import argparse
import os
import sys
from six.moves import shlex_quote
from constants import *

# TODO:add argument parser
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-e', '--env-id', type=str, default="seekavoid_arena_01",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/unreal",
                    help="Log directory path")
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")
# send command to tmux session for worker
def new_cmd(sess, name, cmd):
  if isinstance(cmd, (list, tuple)):
    cmd = " ".join(shlex_quote(str(v)) for v in cmd)
  command = "tmux send-keys -t {}:{} {} Enter".format(sess, name, shlex_quote(cmd))
  return name, command

# create command for launching TF workers
def create_cmd(sess, num_workers, env_id, logdir, visualise=False):
  base_cmd = [
    'CUDA_VISIBLE_DEVICES=',
    sys.executable, 'worker.py',
    '--log-dir', logdir,
    '--env-id', env_id,
    '--num-workers', str(num_workers)
  ]

  if visualise:
    base_cmd = base_cmd + '--visualise'
  
  # parameter-server command
  cmds_map = [new_cmd(sess, 'ps', base_cmd+['--job-name','ps'])]
  # worker command
  for i in range(num_workers):
    cmds_map += [new_cmd(sess, 'worker%d' % i, 
      base_cmd+['--job-name','worker','--task',str(i)])]
  # tensorboard command
  cmds_map += [new_cmd(sess, 'tb', ['tensorboard','--logdir',logdir])]
  # htop command
  cmds_map += [new_cmd(sess, 'htop', ['htop'])]

  windows = [v[0] for v in cmds_map]
  notes = []
  cmds = [
    'mkdir -p {}'.format(logdir),
    'echo {} {} > {}/cmd.sh'.format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir)
  ]
  notes += ['Use `tmux attach -t {}` to watch process output'.format(sess)]
  cmds += [
    'kill $( lsof -i:6006 -t ) > /dev/null 2>&1', # kill any process using the tb port
    'kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1'.format(num_workers+12222), # kill any process using the ps/worker port
    'tmux kill-session -t {}'.format(sess),
    'tmux new-session -s {} -n {} -d bash'.format(sess, windows[0])
  ]
  for w in windows[1:]:
    cmds += ['tmux new-window -t {} -n {} bash'.format(sess, w)]
  cmds += ['sleep 1']

  for window, cmd in cmds_map:
    cmds += [cmd]
  
  return cmds, notes

def run():
  args = parser.parse_args()
  #cmds, notes = create_cmd('unreal', args.num_workers, args.env_id, args.logdir, args.visualise)
  cmds, notes = create_cmd('unreal', num_workers=PARALLEL_SIZE, env_id=ENV_NAME, logdir=LOG_FILE)
  print('Executing the following commands:')
  print('\n'.join(cmds))
  print('')
  os.system('\n'.join(cmds))
  print('\n'.join(notes))

if __name__ == '__main__':
  run()