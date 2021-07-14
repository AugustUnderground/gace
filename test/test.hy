(import logging)
(import [icecream [ic]])
(import [numpy :as np])

(import gym)
(import [stable-baselines3.common.env-checker [check-env]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv env (gym.make "gym_ad:symmetrical-amplifier-v0" 
                    :nmos-prefix "./models/90nm-nmos"
                    :pmos-prefix "./models/90nm-pmos"
                    :lib-path "./libs/90nm_bulk.lib"))

(check-env env :warn True :skip-render-check False)

(setv obs0 (env.reset))
(setv (, obs rew don inf) (env.step (env.action-space.sample)))
(env.render "ascii")
