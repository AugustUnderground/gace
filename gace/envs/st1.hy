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
(import  [typing [List Set Dict Tuple Optional Union]])
(import  [hy.contrib.sequences [Sequence end-sequence]])
(import  [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass ST1V1Env [ACE]
  """
  Base class for geometric design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; The action space consists of 6 parameters ∈ [-1;1]. Each width of the
    ;; schmitt trigger: ["Wp0" "Wn0" "Wp2" "Wp1" "Wn2" "Wn1"]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 6) 
                                 :dtype np.float32)
          self.action-scale-min (np.array (list (repeat self.w-min 6)))
          self.action-scale-max (np.array (list (repeat self.w-max 6))))

    ;; Parent constructor for initialization
    (.__init__ (super ST1V1Env self) #** kwargs))

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wn0 Wn1 Wn2 Wp0 Wp1 Wp2) (unscale-value action 
                                                     self.action-scale-min 
                                                     self.action-scale-max)
          
          sizing {"Wn0" Wn0 "Wn1" Wn1 "Wn2" Wn2 
                  "Wp0" Wp0 "Wp1" Wp1 "Wp2" Wp2}]

      (self.size-circuit sizing))))

(defclass ST1XH035V1Env [ST1V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_id" "st1" "ace_backend" "xh035-3V3" 
                          "variant" 1 "obs_shape" (, 12)}))))

(defclass ST1SKY130V1Env [ST1V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_id" "st1" "ace_backend" "sky130-1V8" 
                          "variant" 1 "obs_shape" (, 12)}))))
