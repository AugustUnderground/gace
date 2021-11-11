import os
import gym
from gace import check_env

HOME = os.path.expanduser('~')

op = 'op1'

env = gym.make( f'gace:{op}-xh035-v1')
              , ckt_path      = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3/{op}'
              , pdk_path      = f'/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
              , random_target = True
              , )

check_env(env)
