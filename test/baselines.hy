(import os)
(import logging)
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [icecream [ic]])
(import [numpy :as np])
(import [h5py :as h5])
(import gym)
(import [stable-baselines3.common.envs [SimpleMultiObsEnv]])
(import [stable-baselines3.common.vec-env [DummyVecEnv VecNormalize SubprocVecEnv]])
(import [stable-baselines3.common.noise [NormalActionNoise 
                                         OrnsteinUhlenbeckActionNoise]])
(import [stable-baselines3.common.buffers [ReplayBuffer]]) 
(import [stable-baselines3 [A2C TD3 SAC DDPG PPO 
                            HerReplayBuffer ]])
(import [sb3-contrib [QRDQN TQC]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      algorithm-name f"a2c"
      model-path f"./models/baselines/{algorithm-name}-miller-amp-xh035-{time-stamp}.mod"
      ;model-path f"./models/baselines/a2c-sym-amp-xh035-{time-stamp}.mod"
      ;model-path f"./models/baselines/a2c-sym-amp-xh035-100456-210903.mod"
      data-path  f"../data/symamp/xh035")

(setv nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos"
      pdk-path  f"{HOME}/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      jar-path  f"{HOME}/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
      moa-path  f"../library/moa"
      sym-path  f"../library/sym"
      tech-cfg  f"../library/techdef/xh035.yaml"
      env1-name "gym_ad:miller-amp-xh035-v0"
      env2-name "gym_ad:sym-amp-xh035-v0")

;; Create Environment
(setv env (gym.make env2-name
                    :nmos-path       nmos-path
                    :pmos-path       pmos-path
                    :pdk-path        pdk-path
                    :ckt-path        sym-path
                    :tech-cfg        tech-cfg
                    :data-log-prefix data-path
                    :close-target    True))

;; Vectorize for normalization
(setv denv (DummyVecEnv [#%(identity env)]))       ; just 1 env
(setv venv (DummyVecEnv (list (repeat #%(identity env) 64))))  ; n envs
(setv dnenv (VecNormalize denv :training True :norm-obs True :norm-reward True))
(setv nenv (VecNormalize venv :training True :norm-obs True :norm-reward True))

;; Add some Gaussian Noise
;(setv n-actions (get nenv.action-space.shape -1)
;      action-noise (NormalActionNoise :mean (np.zeros n-actions) 
;                                      :sigma (* 0.1 (np.ones n-actions))))

;; Model
;(setv model (PPO "MlpPolicy" nenv :verbose 1))
(setv model (A2C "MlpPolicy" nenv :verbose 1 :tensorboard-log "../logs/baselines3"))
;(setv model (TQC "MlpPolicy" nenv :verbose 1))
;(setv model (SAC "MlpPolicy" dnenv
;              :verbose 1
;              :buffer-size (int 1e6)
;              :learning-rate 1e-3
;              :gamma 0.95
;              :batch-size 32
;              :policy-kwargs (dict :net-arch [256 128 64])))

;; Train and save
(model.learn :total-timesteps 10000 :log-interval 1)
(model.save model-path)

;; Store char data
;(setv ts (-> datetime (. datetime) (.now) (.strftime "%y%m%d-%H%M%S")))
;(with [h5-file (h5.File f"../data/symamp/{ts}.h5" "a")]
;  (for [col env.data-log]
;    (setv (get h5-file col) (.to-numpy (get env.data-log col)))))

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
