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

(defclass OP5Env [ACE]
  """
  Base class for OP5
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5Env self) #** (| kwargs {"ace_id" "op5"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 gmid-ls1 gmid-ref
             fug-cm1  fug-cm2  fug-cm3  fug-dp1  fug-ls1  fug-ref
             i1 i2 i3 i4) (unscale-value action self.action-scale-min 
                                                self.action-scale-max)

          i0  (get self.design-constraints "i0" "init")
          vdd (get self.design-constraints "vsup" "init")

          M1 (-> (/ i0     i1) (Fraction) (.limit-denominator 16))
          M2 (-> (/ i0 2.0 i2) (Fraction) (.limit-denominator 20))
          M3 (-> (/ i0 2.0 i3) (Fraction) (.limit-denominator 20))
          M4 (-> (/ i0     i4) (Fraction) (.limit-denominator 3))
          M5 (-> (/ i3     i2) (Fraction) (.limit-denominator 10))

          Mcm11  M1.numerator   Mcm12  M4.denominator Mcm13 M1.denominator
          Mcm31  M5.numerator   Mcm32  M5.denominator
          Mcm212 M2.denominator Mcm222 M3.denominator Mcm2x1 2 
          Mls11  Mcm31          Mls12  Mcm32
          Mdp1   2

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vdd 2) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vdd 2) 0.0]])
          ls1-in (np.array [[gmid-ls1 fug-ls1 (/ vdd 2) 0.0]])
          ref-in (np.array [[gmid-ref fug-ref (/ vdd 2) 0.0]])

          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cm3-out (first (self.nmos.predict cm3-in))
          ls1-out (first (self.pmos.predict ls1-in))
          ref-out (first (self.pmos.predict ref-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Lls1 (get ls1-out 1)
          Lref (get ref-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i0     (get cm1-out 0))
          Wcm2 (/ i1 2.0 (get cm2-out 0))
          Wcm3 (/ i3     (get cm3-out 0))
          Wls1 (/ i3     (get ls1-out 0)) 
          Wref (/ i4     (get ref-out 0)) 

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Lc1"  Lls1  "Lr" Lref
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wc1"  Wls1  "Wr" Wref
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm212" Mcm212 "Mcm31" Mcm31 "Mc11" Mls11
                             "Mcm12" Mcm12 "Mcm222" Mcm222 "Mcm32" Mcm32 "Mc12" Mls12
                             "Mcm13" Mcm13 "Mcm2x1" Mcm2x1
                  #_/ }]

    (self.size-circuit sizing)))

  (defn step-v1 [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Ldp1 Lcm1  Lcm2   Lcm3  Lls1  Lref 
             Wdp1 Wcm1  Wcm2   Wcm3  Wls1  Wref 
                  Mcm11 Mcm212 Mcm31 Mls11 
                  Mcm12 Mcm222 Mcm32 Mls12
                  Mcm13 Mcm2x1) (unscale-value action self.action-scale-min 
                                               self.action-scale-max)

          Mdp1 2

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Lc1"  Lls1  "Lr" Lref
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wc1"  Wls1  "Wr" Wref
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm212" Mcm212 "Mcm31" Mcm31 "Mc11" Mls11
                             "Mcm12" Mcm12 "Mcm222" Mcm222 "Mcm32" Mcm32 "Mc12" Mls12
                             "Mcm13" Mcm13 "Mcm2x1" Mcm2x1
                  #_/ }]

      (self.size-circuit sizing))))

(defclass OP5XH035V0Env [OP5Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))

(defclass OP5XH035V1Env [OP5Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP5XH018V0Env [OP5Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))

(defclass OP5XH018V1Env [OP5Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP5SKY130V0Env [OP5Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5SKY130V0Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 0}))))

(defclass OP5SKY130V1Env [OP5Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5SKY130V1Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass OP5GPDK180V0Env [OP5Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))

(defclass OP5GPDK180V1Env [OP5Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))
