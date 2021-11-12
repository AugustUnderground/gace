(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import gym)
(import [gym.spaces [Dict Box Discrete MultiDiscrete Tuple]])

(import [.ace [ACE]])
(import [gace.util.func [*]])
(import [gace.util.target [*]])
(import [gace.util.render [*]])
(import [hace :as ac])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass NAND4Env [ACE]

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] 
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; ACE ID, required by parent
    (setv self.ace-id "nand4")

    ;; Call Parent Contructor
    (.__init__ (super NAND4Env self) max-steps target random-target noisy-target 
                                     data-log-path param-log-path)

    ;; ACE setup
    (setv self.ace-constructor (ace-constructor self.ace-id self.ace-backend 
                                                :ckt ckt-path :pdk [pdk-path])
          self.ace (self.ace-constructor))

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, 12)  :dtype np.float32))))

(defclass NAND4V1Env [NAND4Env]

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] 
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; Parent constructor for initialization
    (.__init__ (super NAND4V1Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps 
               :data-log-path data-log-path :param-log-path param-log-path)

    ;; The action space consists of 5 parameters ∈ [-1;1]. Each width of the
    ;; inverter chain:  ['wn0', 'wp', 'wn2', 'wn1', 'wn3']
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 5) 
                                 :dtype np.float32)
          self.action-scale-min (np.array (list (repeat self.w-min 5)))
          self.action-scale-max (np.array (list (repeat self.w-max 5))))
    #_/ )

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wn0 Wn1 Wn2 Wn3 Wp1) (unscale-value action self.action-scale-min 
                                                        self.action-scale-max)
          
          sizing {"Wn0" Wn0 "Wn1" Wn1 "Wn2" Wn2 "Wn3" Wn3 "Wp" Wp1}]

      (self.size-circuit sizing))))

(defclass NAND4XH035V1Env [NAND4V1Env]

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^float [reltol 1e-3]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    (setv self.ace-backend "xh035-3V3"
          self.reltol reltol)

    (for [(, k v) (-> self.ace-backend (technology-data) (.items))]
      (setattr self k v))

    (.__init__ (super NAND4XH035V1Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps 
               :data-log-path data-log-path :param-log-path param-log-path)))
