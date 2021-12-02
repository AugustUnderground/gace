import gym
import numpy as np
from gace import check_env
from gace.util.func import target_distance
from typing import Dict, Callable

def reward ( performance: Dict[str, float]
           , target: Dict[str, float]
           , condition: Dict[str, Callable]
           , tolerance: float) -> float:
    loss, mask, _, _ = target_distance(performance, target, condition)
    cost = (( ( -(loss ** 2.0)) / 2.0 / tolerance) * mask
           + (-np.abs(loss) - (0.5 * tolerance)) * np.invert(mask))
    return float(np.sum(np.nan_to_num(cost)))

env = gym.make('gace:op2-xh035-v1', custom_reward = reward)

check_env(env)
