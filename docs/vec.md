## Vectorized Environments (WIP)

**This is still in early development and might not function properly.**

For vectorization / parallel computing
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html) can't
be used, because of the way GAC²E communicates with AC²E, and the Java stuff
not being picklable. That's why `gym.vector.make` wont work. **However**, AC²E
comes with it's own `EnvironmentPool`! GAC²E makes use of this and provdes the
`gace.VecACE` constructor.

#### Example

```python
import gym
import gace

n = 64
env_id = "gace:op2-xh035-v0"
envs = gace.VecACE(env_id, n)
obs = envs.reset()

tic = time.time()
actions = [as.sample() for as in envs.action_space]
obs, rew, don, inf = envs.step(actions)
toc = time.time()
print(f"Evaluating {n} envs took {(toc - tic):.4}s => ~ ca. {(n / (toc - tic)):.3} FPS.")
```

This code can also be found in `examples/vec.py`.

