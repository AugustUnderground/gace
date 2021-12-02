(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

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

(defclass OP3Env [ACE]
  """
  Base class for OP3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3Env self) #** (| kwargs {"ace_id" "op3"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
             fug-cm1  fug-cm2  fug-cm3  fug-dp1 
             i1 i2 i3 ) (unscale-value action self.action-scale-min 
                                              self.action-scale-max)

          i0  (get self.design-constraints "i0"   "init")
          vdd (get self.design-constraints "vsup" "init")

          M1-lim (get self.design-constraints "Mcm12"  "max")
          M2-lim (get self.design-constraints "Mcm212" "max")
          M3-lim (get self.design-constraints "Mcm222" "max")
          M4-lim (get self.design-constraints "Mcm32"  "max")

          M1 (-> (/ i0     i1) (Fraction) (.limit-denominator M1-lim))
          M2 (-> (/ i1 2.0 i2) (Fraction) (.limit-denominator M2-lim))
          M3 (-> (/ i1 2.0 i3) (Fraction) (.limit-denominator M3-lim))
          M4 (-> (/ i2     i3) (Fraction) (.limit-denominator M4-lim))

          Mcm11  M1.numerator Mcm12 M1.denominator
          Mcm31  M4.numerator Mcm32 M4.denominator
          Mcm212 (round (/ i1 i3))  Mcm222 (round (/ i1 i2))
          Mcm2x1 (get self.design-constraints "Mcm2x1" "init") 
          Mdp1   (get self.design-constraints "Md"     "init")

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vdd 2) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vdd 2) 0.0]])

          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cm3-out (first (self.nmos.predict cm3-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i0     (get cm1-out 0))
          Wcm2 (/ i1 2.0 (get cm2-out 0))
          Wcm3 (/ i2     (get cm3-out 0))

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3 
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3 
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm2x1" Mcm2x1 "Mcm31" Mcm31 
                             "Mcm12" Mcm12 "Mcm212" Mcm212 "Mcm32" Mcm32 
                                           "Mcm222" Mcm222 
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP3XH035V0Env [OP3Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))

(defclass OP3XH035V1Env [OP3Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP3XH035V3Env [OP3Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH035V3Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 3}))))

(defclass OP3XH018V0Env [OP3Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))

(defclass OP3XH018V1Env [OP3Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP3XH018V3Env [OP3Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH018V3Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 3}))))

(defclass OP3XT018V0Env [OP3Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XT018V0Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 0}))))

(defclass OP3XT018V1Env [OP3Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XT018V1Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass OP3XT018V3Env [OP3Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XT018V3Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 3}))))

(defclass OP3SKY130V0Env [OP3Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3SKY130V0Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 0}))))

(defclass OP3SKY130V1Env [OP3Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3SKY130V1Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass OP3SKY130V3Env [OP3Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3SKY130V3Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 3}))))

(defclass OP3GPDK180V0Env [OP3Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))

(defclass OP3GPDK180V1Env [OP3Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))

(defclass OP3GPDK180V3Env [OP3Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3GPDK180V3Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 3}))))
