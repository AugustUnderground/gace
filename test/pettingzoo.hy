(import requests)
(import os)
(import logging)
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [icecream [ic]])
(import [numpy :as np])
(import [h5py :as h5])
(import gym)
(import [operator [itemgetter]])
(import [stable-baselines3.common.env-checker [check-env]])
(import [gym-ad.envs [SymAmpXH035MA sym-env sym-raw]])
(import [pettingzoo.test [api-test]])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      algorithm-name f"marl"
      model-path f"./models/baselines/{algorithm-name}-sym-amp-xh035-{time-stamp}.mod"
      data-path  f"../data/symamp/xh035"
      nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos")

(setv env (sym-env :nmos-path       nmos-path
                   :pmos-path       pmos-path
                   :data-log-prefix data-path
                   :close-target    True))

(api-test env :num-cycles 10 :verbose-progress True)
