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
(import [pyarrow [parquet :as pq]])
(import [hace :as ac])
(import gym)
(import gace)
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv env (gym.make "gace:op2-xh035-v0"))
(setv obs (.reset env))

(setv env (gym.make "gace:op2-xh035-v2")); :target-filter ["a_0" "ugbw"]))
(setv obs (.reset env))
(setv (, o r d i) (.random-step env)) (pp o)

(for [i (range 50)] 
  (setv (, o r d _) (env.random-step))
  (print f"{env.num-steps}: {d} -> {r}")
  (when d 
    (print "RESET")
    (env.reset)))


(for [i (range 10)] (setv (, o r d _) (env.random-step)) (print f"{i}: {d} -> {r}"))

(setv n 5)
(setv envs (gace.vector-make-same "gace:op2-xh035-v0" n :target-filter ["a_0" "ugbw"])) 

(setv n 5)
(setv envs (gace.vector-make-same "gace:op2-xh035-v2" n)) 
(setv obs (.reset envs))
(for [i (range 10)] (envs.reset) (pp (ac.current-sizing (. (first envs) ace))) (for [i (range 10)] (envs.random-step)))

(for [i (range 10)] (setv (, o r d _) (envs.random-step)) (print f"{i}: {d} -> {r}"))

(for [i (range 10)] (env.reset) (pp (ac.current-sizing env.ace)) (for [i (range 10)] (env.random-step)))


(setv n 5)
(setv envs (gace.vector-make-same "gace:op2-xh035-v0" n :sim-path "/dev/null")) 
;(setv envs (gace.vector-make-same "gace:nand4-xh035-v1" n)) 
(setv obs (.reset envs))
;(setv obs (.reset envs [7 3 29]))

(for [i (range 105)] (setv (, o0 r d _) (envs.random-step)) (print f"{i}: {d} -> {r}"))

(first o0)
(second o0)

(setv o1 (envs.reset :done-mask d))

(first o1)
(second o1)

(setv o2 (envs.reset :done-mask [True False True False False]))

(first o2)
(second o2)

(get o0 1)
(get o1 1)
(get o2 1)

(get o0 2)
(get o1 2)
(get o2 2)

(for [_ (range 10)] (envs.step (dfor (, i a) (enumerate envs.action-space) [i (a.sample)])))


(setv env (get (list envs) 3))

(get env.data-log "performance")







(setv env (gym.make "gace:op2-xh035-v0" :sim-path "/dev/null/"))
(setv o (env.reset))
(setv (, o r d i) (env.random-step))
(setv sizing (ac.initial-sizing env.ace))

(setx act (env.action-space.sample))
(setv (, o r d i) (env.step act))





(setv env (gym.make "gace:nand4-xh035-v1" :reltol 0.01 :noisy-target False :random-target False))
(env.reset)
;; Wp=10e-6 Wn0=3.755e-6 Wn1=6.6e-06 Wn2=12.0e-06 Wn3=19.65e-06
(setx action (gace.scale-value (np.array [3.755e-6 10e-6 12.0e-6 6.6e-6 19.65e-6]) env.action-scale-min env.action-scale-max))


(setv (, o r d i) (env.step action))








(setv p (ac.current-performance env.ace))

(pp (setx ra (dfor k (get env.info "actions") 
    [k (cond [(.endswith k ":fug") (np.log10 (get p k))]
             [(.endswith k ":id") (* (get p k) 1e6)]
             [True (get p k)])])))

(setx sa (gace.scale-value (np.array (list (.values ra))) env.action-scale-min env.action-scale-max))

(setv (, o r d i) (env.step sa))

(setv p1 (ac.current-performance env.ace))
(setx ra1 (dfor k (get env.info "actions") [k (get p1 k)]))

(pp (setx ra1 (dfor k (get env.info "actions") 
    [k (cond [(.endswith k ":fug") (np.log10 (get p1 k))]
             [(.endswith k ":id") (* (get p1 k) 1e6)]
             [True (get p1 k)])])))

(pp ra)
(pp ra1)

(pp (dfor k (.keys ra) [k (round (abs (/ (- (get ra k) (get ra1 k)) (get ra k))) :ndigits 2)] ))

(setv vs ["MNCM11:vgs" "MNCM11:vds" "MNCM11:vbs" 
       "MPCM221:vgs" "MPCM221:vds" "MPCM221:vbs" 
       "MNCM31:vgs" "MNCM31:vds" "MNCM31:vbs"
       "MND11:vgs" "MND11:vds" "MND11:vbs"])

(pp (setx va (dfor v vs [v (get p1 v)])))


(setv (, idoverw l gdsoverw vgs) (.tolist (.squeeze (env.pmos.predict (np.array [[ 9.97 (np.power 10 7.645) 1.0 0.0 ]])))))

(setv ids ["MNCM11:id" "MPCM221:id" "MNCM31:id" "MND11:id"])
(pp (setx ia (dfor i ids [i (get p1 i)])))

(setv w (/ 3e-6 idoverw))

(pp  (ac.current-sizing env.ace))







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

