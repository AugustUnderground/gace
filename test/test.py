import gym
from stable_baselines3.common.env_checker import check_env

env = gym.make( "gym_ad:symmetrical-amplifier-v0"
              , nmos_prefix = "../models/90nm-nmos"
              , pmos_prefix = "../models/90nm-pmos"
              , lib_path    = "../libs/90nm_bulk.lib"
              , )

check_env( env
         , warn              = True
         , skip_render_check = False
         , )
