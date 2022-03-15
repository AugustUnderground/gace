import numpy as np
import gym, gace
import hace as ac
from hy.contrib.pprint import pp

env = gym.make( "gace:nand4-xh035-v1", reltol = 0.01
              , noisy_target = False
              , random_target = False
              , )

obs = env.reset()

# Target satisfying action:
# Wp=10e-6 Wn0=3.755e-6 Wn1=6.6e-06 Wn2=12.0e-06 Wn3=19.65e-06
true_action = np.array([3.755e-6, 10e-6, 12.0e-6, 6.6e-6, 19.65e-6])
action = gace.scale_value( true_action
                         , env.action_scale_min
                         , env.action_scale_max
                         , )

o,r,d,i = env.step(action)

obs_env = dict(zip(i["observations"], o))
obs_ace = ac.current_performance(env.ace)

pp({p: (obs_env[p],obs_ace[p]) for p in obs_ace.keys() })

