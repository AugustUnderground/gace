import os
import gym
from stable_baselines3.common.env_checker import check_env

HOME = os.path.expanduser('~')

env = gym.make( f'gym_ad:sym-amp-xh035-v0'
              , nmos_path     = f'../models/xh035-nmos'
              , pmos_path     = f'../models/xh035-pmos'
              , ckt_path      = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3/op2'
              , pdk_path      = f'/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
              , random_target = False
              , )

check_env( env
         , warn              = True
         , skip_render_check = True
         , )
