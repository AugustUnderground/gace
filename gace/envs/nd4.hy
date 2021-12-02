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

(defclass NAND4Env [ACE]
  """
  Base class for NAND4
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4Env self) #** (| kwargs {"ace_id" "nand4"}))))

(defclass NAND4XH035V1Env [NAND4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass NAND4XH035V3Env [NAND4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V3Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 3}))))

(defclass NAND4XH018V1Env [NAND4Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass NAND4XT018V1Env [NAND4Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XT018V1Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass NAND4SKY130V1Env [NAND4Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4SKY130V1Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass NAND4GPDK180V1Env [NAND4Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))
