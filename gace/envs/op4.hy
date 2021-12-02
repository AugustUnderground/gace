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

(defclass OP4Env [ACE]
  """
  Base class for OP4
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4Env self) #** (| kwargs {"ace_id" "op4"})))

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
             i1 i2 i3 ) (unscale-value action self.action-scale-min 
                                              self.action-scale-max)

          i0  (get self.design-constraints "i0"   "init")
          vdd (get self.design-constraints "vsup" "init")

          M1-lim (get self.design-constraints "Mcm13" "max")
          M2-lim (get self.design-constraints "Mcm22" "max")
          M3-lim (get self.design-constraints "Mcm12" "max")

          M1 (-> (/ i0     i1) (Fraction) (.limit-denominator M1-lim))
          M2 (-> (/ i1 2.0 i2) (Fraction) (.limit-denominator M2-lim))
          M3 (-> (/ i0     i3) (Fraction) (.limit-denominator M3-lim))

          Mdp1  (get self.design-constraints "Md" "init")
          Mcm11 M1.numerator Mcm12 M3.denominator Mcm13 M1.denominator
          Mcm21 M2.numerator Mcm22 M2.denominator
          Mcm31 (get self.design-constraints "Mcm31" "init")            
          Mcm32 (get self.design-constraints "Mcm32" "init")
          Mls1  Mcm22

          ;vx (/ self.vdd 2.7)

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
          Wcm3 (/ i2     (get cm3-out 0))
          Wls1 (/ i2     (get ls1-out 0)  Mls1) 
          Wref (/ i3     (get ref-out 0)) 

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3 "Lc1" Lls1 "Lr" Lref
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3 "Wc1" Wls1 "Wr" Wref
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31"Mc1" Mls1 
                             "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32  
                             "Mcm13" Mcm13 
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP4XH035V0Env [OP4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))

(defclass OP4XH035V1Env [OP4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V1Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP4XH018V0Env [OP4Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))

(defclass OP4XH018V1Env [OP4Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH018V1Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP4XT018V0Env [OP4Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XT018V0Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 0}))))

(defclass OP4XT018V1Env [OP4Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XT018V1Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass OP4SKY130V0Env [OP4Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4SKY130V0Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 0}))))

(defclass OP4SKY130V1Env [OP4Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4SKY130V1Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass OP4GPDK180V0Env [OP4Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))

(defclass OP4GPDK180V1Env [OP4Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4GPDK180V1Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))
