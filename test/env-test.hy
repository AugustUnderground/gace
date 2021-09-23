(import os)
(import logging)
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [icecream [ic]])
(import [numpy :as np])
(import [h5py :as h5])
(import gym)
(import [stable-baselines3.common.env-checker [check-env]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      model-path f"./models/baselines/a2c-miller-amp-xh035-{time-stamp}.mod"
      ;model-path f"./models/baselines/a2c-sym-amp-xh035-{time-stamp}.mod"
      ;model-path f"./models/baselines/a2c-sym-amp-xh035-100456-210903.mod"
      data-path  f"../data/symamp/xh035")

(setv nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos"
      pdk-path  f"{HOME}/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      jar-path  f"{HOME}/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
      ckt-path  f"../library/"
      env2-name "gym_ad:sym-amp-xh035-v0"
      env1-name "gym_ad:miller-amp-xh035-v0")

;; Create Environment
(setv env (gym.make env1-name
                    :nmos-path       nmos-path
                    :pmos-path       pmos-path
                    :pdk-path        pdk-path
                    :ckt-path        ckt-path
                    :data-log-prefix data-path
                    :close-target    True))

;; Check if no Warnings
(check-env env :warn True)

;; One step test
(setv obs (.reset env))
(setv act (.sample env.action-space))
(setv ob (.step env act))

;; Ten step test
(for [i (range 10)]
  (setv act (.sample env.action-space))
  (setv act (.sample env.action-space))
  (setv ob (.step env act))
  (pp (get ob 1)))
