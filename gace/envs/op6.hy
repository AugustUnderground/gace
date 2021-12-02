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

(defclass OP6Env [ACE]
  """
  Base class for OP6
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6Env self) #** (| kwargs {"ace_id" "op6"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-cm1 gmid-cm2 gmid-cs1 gmid-dp1 gmid-res gmid-cap
             fug-cm1  fug-cm2  fug-cs1  fug-dp1  fug-res  fug-cap
             i1 i2 ) (unscale-value action self.action-scale-min 
                                           self.action-scale-max)

          i0  (get self.design-constraints "i0"   "init")
          vdd (get self.design-constraints "vsup" "init")

          M1-lim (get self.design-constraints "Mcm12" "max")
          M2-lim (get self.design-constraints "Mcm13" "max")
          
          M1 (-> (/ i0 i1) (Fraction) (.limit-denominator M1-lim))
          M2 (-> (/ i0 i2) (Fraction) (.limit-denominator M2-lim))
          
          Mcm11 M1.numerator Mcm12 M1.denominator Mcm13 M2.denominator
          Mcm21 (get self.design-constraints "Mcm21" "init")
          Mcm22 (get self.design-constraints "Mcm22" "init")
          Mcs1  (get self.design-constraints "Mcs"   "init")
          Mres  (get self.design-constraints "Mr1"   "init")
          Mcap  (get self.design-constraints "Mc1"   "init")
          Mdp1  (get self.design-constraints "Md"    "init")

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vdd 2.0) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vdd 2.0) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vdd 2.0) 0.0]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (/ vdd 2.0) 0.0]])
          cap-in (np.array [[gmid-cap fug-cap        0.0  0.0]])
          res-in (np.array [[gmid-res fug-res        0.0  0.0]])
          
          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cs1-out (first (self.pmos.predict cs1-in))
          cap-out (first (self.pmos.predict cap-in))
          res-out (first (self.pmos.predict res-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcs1 (get cs1-out 1)
          Lcap (get cap-out 1)
          Lres (get res-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i0     (get cm1-out 0))
          Wcm2 (/ i1 2.0 (get cm2-out 0))
          Wcs1 (/ i2     (get cs1-out 0))
          Wcap (/ i2     (get cap-out 0)) 
          Wres (/ i2     (get res-out 0)) 
          
          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lr1" Lres "Lc1" Lcap "Lcs" Lcs1  
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs" Wcs1 "Wr1" Wres "Wc1" Wcap
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcs" Mcs1 "Mr1" Mres "Mc1" Mcap
                             "Mcm12" Mcm12 "Mcm22" Mcm22
                             "Mcm13" Mcm13 
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP6XH035V0Env [OP6Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))

(defclass OP6XH035V1Env [OP6Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP6XH018V0Env [OP6Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))

(defclass OP6XH018V1Env [OP6Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP6XT018V0Env [OP6Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XT018V0Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 0}))))

(defclass OP6XT018V1Env [OP6Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XT018V1Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass OP6SKY130V0Env [OP6Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6SKY130V0Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 0}))))

(defclass OP6SKY130V1Env [OP6Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6SKY130V1Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass OP6GPDK180V0Env [OP6Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))

(defclass OP6GPDK180V1Env [OP6Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))
