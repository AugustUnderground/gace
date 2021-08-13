(import os)
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

(setv HOME (os.path.expanduser "~"))

;; Create Environment
(setv env (gym.make "gym_ad:sym-amp-xh035-v0" 
                    :nmos-path f"./models/xh035-nmos"
                    :pmos-path f"./models/xh035-pmos"
                    :pdk-path f"{HOME}/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
                    :jar-path f"{HOME}/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
                    :ckt-path f"./libs"
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
(model.learn :total-timesteps 10000 :log-interval 1)


;; Test
(setv obs (.reset env))
(setv act (.sample env.action-space))
(setv ob (.step env act))




(setv df (pd.DataFrame :columns ["A" "B" "C"]))

(setv dd {"A" 1 "B" 2 "C" 3 "D" 5})

(df.append dd :ignore-index True)




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
