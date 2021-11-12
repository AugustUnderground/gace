(import os)
(import yaml)
(import logging)
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [numpy :as np])
(import [h5py :as h5])
(import [hace :as ac])
(import gym)
(import gace)
(import [operator [itemgetter]])
(import [stable-baselines3.common.env-checker [check-env]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv env (gym.make "gace:op4-xh035-v0"))
(gace.check-env env)



(for [o (range 6)]
  (setv op f"op{(inc o)}")
  (setv env0 (gym.make f"gace:{op}-xh035-v0"))
  (setv env1 (gym.make f"gace:{op}-xh035-v1"))
  (print f"TESTING {op} v0")
  (gace.check-env env0)
  (print f"TESTING {op} v1")
  (gace.check-env env1)
  (del env0) (del env1))

(check-env env :warn True)

(env.reset)

(list (map #%(unscale-value #* %1) (zip foo env.action-scale-min env.action-scale-max)))

(setv (, o r d i) (env.step (np.array bar)))



(setx obs (.reset env))
(setx act (.sample env.action-space))
(setx ob (.step env act))

(for [i (range 10)]
  (setv act (.sample env.action-space))
  (setv ob (.step env act))
  (pp (get ob 1)))

