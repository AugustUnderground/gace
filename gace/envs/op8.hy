(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

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

(defclass OP8Env [ACE]
  """
  Base class for OP8
  """
  (defn __init__ [self &kwargs kwargs]
    (setv self.num-gmid 6
          self.num-fug 6
          self.num-ib 4)
    (.__init__ (super OP8Env self) #** (| kwargs {"ace_id" "op8"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. 
    """
    (let [unscaled-action (unscale-value action self.action-scale-min 
                                                self.action-scale-max)
      
          (, gmid-cm5 gmid-cm4 gmid-cm3 gmid-cm2
             gmid-cm1 gmid-dp1)                   (as-> unscaled-action it
                                                      (get it (slice None 6)))
          (, fug-cm5  fug-cm4  fug-cm3  fug-cm2
             fug-cm1  fug-dp1)                    (as-> unscaled-action it
                                                      (get it (slice 6 12))
                                                      (np.power 10 it))
          (, i1 i2 i3 i4)                         (as-> unscaled-action it
                                                      (get it (slice -4 None))
                                                      (np.array it)
                                                      (* it 1e-6))

          i0  (get self.design-constraints "i0"   "init")
          vdd (get self.design-constraints "vsup" "init")
          
          M1-lim (-> self (. design-constraints) (get "Mcm53" "max") (int))
          M2-lim (-> self (. design-constraints) (get "Mcm52" "max") (int))
          M3-lim (-> self (. design-constraints) (get "Mcm42" "max") (int))
          M4-lim (-> self (. design-constraints) (get "Mcm43" "max") (int))

          M1 (-> (/ i0 i1) (Fraction) (.limit-denominator M1-lim))
          M2 (-> (/ i0 i2) (Fraction) (.limit-denominator M2-lim))
          M3 (-> (/ i2 i3) (Fraction) (.limit-denominator M3-lim))
          M4 (-> (/ i2 i4) (Fraction) (.limit-denominator M4-lim))

          Mdp1 (get self.design-constraints "Md1"  "init") 
          Mcm1 (get self.design-constraints "Mcm1" "init") 
          Mcm2 (get self.design-constraints "Mcm2" "init") 
          Mcm3 (get self.design-constraints "Mcm3" "init")

          Mcm51 M1.numerator Mcm52 M2.denominator Mcm53 M1.denominator
          Mcm41 M3.numerator Mcm42 M3.denominator Mcm43 M4.denominator

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vdd 2.0) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vdd 2.0) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vdd 2.0) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vdd 2.0) 0.0]])
          cm4-in (np.array [[gmid-cm4 fug-cm4 (/ vdd 2.0) 0.0]])
          cm5-in (np.array [[gmid-cm5 fug-cm5 (/ vdd 2.0) 0.0]])

          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.nmos.predict cm2-in))
          cm3-out (first (self.pmos.predict cm3-in))
          cm4-out (first (self.pmos.predict cm4-in))
          cm5-out (first (self.nmos.predict cm5-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Lcm4 (get cm4-out 1)
          Lcm5 (get cm5-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i3     (get cm1-out 0))
          Wcm2 (/ i3     (get cm2-out 0))
          Wcm3 (/ i3     (get cm3-out 0))
          Wcm4 (/ i2     (get cm4-out 0))
          Wcm5 (/ i0     (get cm5-out 0)) ]
    
    (setv self.last-action (->> unscaled-action (zip self.input-parameters) (dict)))

    { "Ld1" Ldp1 "Lcm1" Lcm1 "Lcm2" Lcm2 "Lcm3" Lcm3 "Lcm4"  Lcm4  "Lcm5"  Lcm5
      "Wd1" Wdp1 "Wcm1" Wcm1 "Wcm2" Wcm2 "Wcm3" Wcm3 "Wcm4"  Wcm4  "Wcm5"  Wcm5
      "Md1" Mdp1 "Mcm1" Mcm1 "Mcm2" Mcm2 "Mcm3" Mcm3 "Mcm41" Mcm41 "Mcm51" Mcm51
                                                     "Mcm42" Mcm42 "Mcm52" Mcm52
                                                     "Mcm43" Mcm42 "Mcm53" Mcm53 
      #_/ })))

(defclass OP8XH035V0Env [OP8Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))
  
(defclass OP8XH035V1Env [OP8Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP8XH035V3Env [OP8Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH035V3Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 3}))))

(defclass OP8XH018V0Env [OP8Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))
  
(defclass OP8XH018V1Env [OP8Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP8XH018V3Env [OP8Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH018V3Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 3}))))

(defclass OP8XT018V0Env [OP8Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XT018V0Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 0}))))
  
(defclass OP8XT018V1Env [OP8Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XT018V1Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass OP8XT018V3Env [OP8Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XT018V3Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 3}))))

(defclass OP8GPDK180V0Env [OP8Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))
  
(defclass OP8GPDK180V1Env [OP8Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))

(defclass OP8GPDK180V3Env [OP8Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8GPDK180V3Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 3}))))
