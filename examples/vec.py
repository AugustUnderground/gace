import time
import gym
import gace

n = 64
env_id = "gace:op2-xh035-v0"
envs = gace.vector_make_same(env_id, n)
obs = envs.reset()

tic = time.time()
actions = [a.sample() for a in envs.action_space]
obs, rew, don, inf = envs.step(actions)
toc = time.time()

print(f"Evaluating {n} envs took {(toc - tic):.4}s.")
print(f"ca. {(n / (toc - tic)):.3} FPS.")
