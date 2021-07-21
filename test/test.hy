(import logging)
(import [icecream [ic]])
(import [numpy :as np])
(import gym)
(import [stable-baselines3.common.envs [SimpleMultiObsEnv]])
(import [stable-baselines3.common.env-checker [check-env]])
(import [stable-baselines3.common.vec-env [DummyVecEnv VecNormalize]])
(import [stable-baselines3.common.noise [NormalActionNoise 
                                         OrnsteinUhlenbeckActionNoise]])
(import [stable-baselines3.common.buffers [ReplayBuffer]]) 
(import [stable-baselines3 [A2C TD3 SAC DDPG 
                            HerReplayBuffer ]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv oe (SimpleMultiObsEnv :random-start False))
(setv o (.reset oe))
()

;; Create Environment
(setv env (gym.make "gym_ad:symmetrical-amplifier-v0" 
                    :nmos-prefix "./models/90nm-nmos"
                    :pmos-prefix "./models/90nm-pmos"
                    :lib-path "./libs/90nm_bulk.lib"
                    :close-target True))

;; Check if no Warnings
(check-env env :warn True)

;; Vectorize for normalization
(setv venv (DummyVecEnv [#%(identity env)]))
(setv nenv (VecNormalize venv :training True
                              :norm-obs True
                              :norm-reward True))

;; Add some Gaussian Noise
(setv n-actions (get nenv.action-space.shape -1)
      action-noise (NormalActionNoise :mean (np.zeros n-actions) 
                                      :sigma (* 0.1 (np.ones n-actions))))

;; Model
(setv model (TD3 "MlpPolicy" nenv :action-noise action-noise :verbose 1))

;; Train
(model.learn :total-timesteps 100 :log-interval 1)








(model.save "./models/basleline/td3_symamp90.mod")

(setv mod-env (.get-env model))

(del model)

(setv model (TD3.load "./models/basleline/td3_symamp90.mod"))

(loop [[i 0] [obs (env.reset)]]
  (setv (, action state)    (model.predict obs :deterministic True)
        (, obs reward done info) (env.step action))
  (print f"[{i :04}] Reward: {reward}")
  (if (> i 100)
      (env.render "bode")
      (recur (inc i) (if done (env.reset) obs))))
