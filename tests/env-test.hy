(import os)
(import time)
(import yaml)
(import logging)
(import [functools [partial]])
(import [fractions [Fraction]])
(import [datetime [datetime :as dt]])
(import [numpy :as np])
(import [h5py :as h5])
(import [pyarrow :as pa])
(import [pyarrow [feather :as ft]])
(import [hace :as ac])
(import gym)
(import gace)
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv env (gym.make "gace:op2-xh035-v0"))
(env.reset)
(setv (, o r d i) (env.random-step))
(setv sizing (ac.initial-sizing env.ace))
(len (env.action-space.sample))


(setv dt1 (pa.table (list (repeat (pa.array (.tolist (np.random.rand 10)) :type (.float32 pa)) 7)) :names ["a" "b" "c" "d" "e" "f" "g"]))

(setv dt2 (pa.table (list (repeat (pa.array (.tolist (np.random.rand 10)) :type (.float32 pa)) 7)) :names ["a" "b" "c" "d" "e" "f" "g"]))


(setv dt3 (-> [dt1 dt2] (pa.concat_tables) (.combine-chunks)))


(+ [(pa.array [1]) (pa.array [2])] [(pa.array [1]) (pa.array [2])])



(setv n 5)
(setv envs (gace.vector-make-same "gace:op2-xh035-v0" n)) 
;(setv envs (gace.vector-make-same "gace:nand4-xh035-v1" n)) 
(setv obs (.reset envs))
;(setv obs (.reset envs [7 3 29]))
(setv (, o r _ _) (envs.random-step))

(dfor (, i e) (enumerate envs) [i (ac.current-performance e.ace)])


(get envs.info 0 "output-parameters")

(lfor e envs.info (len (get e "output-parameters")))

(list (zip #* (lfor e envs (, (list (.keys e.target)) e.input-parameters))))


(for [_ (range 10)]
  (setv (, o r _ _) (envs.random-step))
  (pp r))


(pp (setx actions (lfor e envs (e.action-space.sample))))

(envs.step actions)

(setv a (* (np.ones 10) 2.0))
(setv (, o r _ _) (env.step a))


(unscale-value (env.action-space.sample) env.action-scale-min env.action-scale-max)

(setv action (np.hstack (, (np.array [10.0 10.0 10.0 10.0]) 
                           (np.power 10 (np.array [8.0 8.0 8.0 8.0]))
                           (np.array [3e-6 12e-6]))))

(env.unscale-action (env.scale-action action))

(env.action-space.sample)
(setv (, _ r _ _) (env.step (env.action-space.sample)))

(-> (/ 3e-6 1.16e-5) (Fraction) (.limit-denominator 16))

(setv (, o r d i) (env.size-circuit sizing))
(setv (, o r d i) (env.size-circuit (| sizing {"Wn0" 4.7e-6})))

(setv env (gym.make "gace:op2-xh035-v0"))
(env.reset)
(setv sizing (ac.initial-sizing env.ace))

(setv (, o r d i) (env.size-circuit sizing))
(setv (, o r d i) (env.size-circuit (| sizing {"Wd" 3.5e-6})))


(-> env (. ace) (ac.current-performance) (get "a_0"))







(setx sizing (ac.random-sizing env.ace))
(env.size-circuit sizing)

(dfor (, i e) (enumerate (get i "output-parameters")) :if (.startswith e "performance") [e (get o i)])

(dict (env.unscaled-sample))

(setv obs (.reset env))
(gace.check-env env)

(setx act (.sample env.action-space))
(pp (.step env act))

(env.random-step)






(setv tic (.time time))
(setv (, obs rew don inf) (envs.step (lfor as envs.action-space (.sample as))))
(setv toc (.time time))
(print f"Evaluating {n} envs took {(- toc tic):.4}s -> {(/ n (- toc tic)):.3} FPS.")


(setv env (gym.make "gace:op2-xh035-v1"))

(setv obs (.reset env))
(gace.check-env env)

(setx act (.sample env.action-space))
(pp (.step env act))

(env.random-step)

(for [i [1 2 3 4 5 6 8 9]]
  (for [e [0 1]]
    (setv env (gym.make f"gace:op{i}-xh035-v{e}"))
    (print f"TESTING op{i} v{e}")
    (gace.check-env env)
    (.close env) 
    (del env)))


(for [o [1 2 3 4 5 6 8 9]]
  (setv op f"op{o}")
  (setv env0 (gym.make f"gace:{op}-gpdk180-v0"))
  ;(setv env1 (gym.make f"gace:{op}-gpdk180-v1"))
  (print f"TESTING {op} v0")
  (gace.check-env env0)
  ;(print f"TESTING {op} v1")
  ;(gace.check-env env1)
  (.close env0) 
  ;(.close env1)
  (del env0) 
  ;(del env1)
  #_/ )

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

