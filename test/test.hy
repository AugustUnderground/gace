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

;; Create Environment
(setv env (gym.make "gym_ad:sym-amp-xh035-v0" 
                    :nmos-path "./models/xh035-nmos"
                    :pmos-path "./models/xh035-pmos"
                    :lib-path "/home/ynk/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
                    :jar-path "/home/ynk/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
                    :ckt-path "./libs"
                    :close-target True))

;; Test
(setv obs (.reset env))

(setv act (.sample env.action-space))

(setv ob (.step env act))


;; Check if no Warnings
(check-env env :warn True)

(setv p (env.nmos.predict [[0.5 0.5 0.5 0.5]]))

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
(model.learn :total-timesteps 10000 :log-interval 1)

(nenv.render "bode")






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
