(import logging)
(import [icecream [ic]])
(import [numpy :as np])
(import gym)
(import [stable-baselines3.common.env-checker [check-env]])
(import [stable-baselines3.common.noise [NormalActionNoise 
                                         OrnsteinUhlenbeckActionNoise]])
(import [stable-baselines3 [A2C TD3]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv env (gym.make "gym_ad:symmetrical-amplifier-v0" 
                    :nmos-prefix "./models/90nm-nmos"
                    :pmos-prefix "./models/90nm-pmos"
                    :lib-path "./libs/90nm_bulk.lib"
                    :close-target True))

(check-env env :warn True)

(setv n-actions (get env.action-space.shape -1)
      action-noise (NormalActionNoise :mean (np.zeros n-actions) 
                                      :sigma (* 0.1 (np.ones n-actions))))

(setv model (TD3 "MlpPolicy" env :action-noise action-noise :verbose 1))

(model.learn :total-timesteps 10000 :log-interval 1)

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
