(import os)
(import logging)
(import pprint)
(import [datetime [datetime :as dt]])
(import [icecream [ic]])
(import [numpy :as np])
(import [h5py :as h5])
(import gym)
(import [stable-baselines3.common.envs [SimpleMultiObsEnv]])
(import [stable-baselines3.common.env-checker [check-env]])
(import [stable-baselines3.common.vec-env [DummyVecEnv 
                                           VecNormalize 
                                           SubprocVecEnv]])
(import [stable-baselines3.common.noise [NormalActionNoise 
                                         OrnsteinUhlenbeckActionNoise]])
(import [stable-baselines3.common.buffers [ReplayBuffer]]) 
(import [stable-baselines3 [A2C TD3 SAC DDPG PPO 
                            HerReplayBuffer ]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv pp         (-> pprint (.PrettyPrinter :indent 2) (. pprint))
      HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      ;model-path f"./models/baselines/a2c-sym-amp-xh035-{time-stamp}.mod"
      model-path f"./models/baselines/a2c-sym-amp-xh035-100456-210903.mod"
      data-path  f"../data/symamp/xh035.h5")

(setv nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos"
      pdk-path  f"{HOME}/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      jar-path  f"{HOME}/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
      ckt-path  f"../library/")

;; Create Environment
(setv env (gym.make "gym_ad:sym-amp-xh035-v0" 
                    :nmos-path      nmos-path
                    :pmos-path      pmos-path
                    :pdk-path       pdk-path
                    :jar-path       jar-path
                    :ckt-path       ckt-path
                    ;:data-log-path  data-path
                    :close-target   True))

;; Check if no Warnings
(check-env env :warn True)

;; Test
;(setv obs (.reset env))
;(setv act (.sample env.action-space))
;(setv ob (.step env act))
;
;(for [i (range 10)]
;  (setv act (.sample env.action-space))
;  (setv ob (.step env act))
;  (pp (get ob 1)))

;; Vectorize for normalization
;(setv venv (DummyVecEnv [#%(identity env)]))
(setv venv (DummyVecEnv (list (repeat #%(identity env) 64))))
(setv nenv (VecNormalize venv :training True
                              :norm-obs True
                              :norm-reward True))

;; Add some Gaussian Noise
(setv n-actions (get nenv.action-space.shape -1)
      action-noise (NormalActionNoise :mean (np.zeros n-actions) 
                                      :sigma (* 0.1 (np.ones n-actions))))

;; Model
;(setv ppo-model   (PPO  "MlpPolicy" nenv :verbose 1))
(setv a2c-model   (A2C  "MlpPolicy" nenv :verbose 1))

;; Train
(a2c-model.learn :total-timesteps 10000 :log-interval 1)

;; Store char data
;(setv ts (-> datetime (. datetime) (.now) (.strftime "%y%m%d-%H%M%S")))
;(with [h5-file (h5.File f"../data/symamp/{ts}.h5" "a")]
;  (for [col env.data-log]
;    (setv (get h5-file col) (.to-numpy (get env.data-log col)))))

(a2c-model.save model-path)

(setv mod-env (.get-env a2c-model))

(del a2c-model)

(setv model (A2C.load model-path))

(loop [[i 0] [obs (nenv.reset)]]
  (setv (, action state)         (model.predict obs :deterministic True)
        (, obs reward done info) (nenv.step action))
  (print f"Done: {done}")
  (print f"[{i :04}] Reward: {reward}")
  (if (> i 100)
      (pp (nenv.performance))
      (recur (inc i) (if done (nenv.reset) obs))))
