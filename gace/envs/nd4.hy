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
(import  [hy.contrib.sequences [Sequence end-sequence]])
(import  [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass NAND4V1Env [ACE]
  """
  Base class for geometric design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]
    
    ;; Parent constructor for initialization
    (.__init__ (super NAND4V1Env self) #** kwargs)

    ;; The action space consists of 5 parameters ∈ [-1;1]. Each width of the
    ;; inverter chain.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 5) 
                                 :dtype np.float32)
          self.action-scale-min (np.array (list (repeat self.w-min 5)))
          self.action-scale-max (np.array (list (repeat self.w-max 5))))

    ;; Specify Input Parameternames
    (setv self.input-parameters ["Wn0" "Wp" "Wn2" "Wn1" "Wn3"]))

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
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_id" "nand4" "ace_backend" "xh035-3V3" 
                          "ace_variant" 1 "obs_shape" (, 12)}))))

(defclass NAND4SKY130V1Env [NAND4V1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4SKY130V1Env self) #**
               (| kwargs {"ace_id" "nand4" "ace_backend" "sky130-1V8" 
                          "ace_variant" 1 "obs_shape" (, 12)}))))

(defclass NAND4GPDK180V1Env [NAND4V1Env]
  """
  Implementation: gpdk180-1V2
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4GPDK180V1Env self) #**
               (| kwargs {"ace_id" "nand4" "ace_backend" "gpdk180-1V2" 
                          "ace_variant" 1 "obs_shape" (, 12)}))))
