(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

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

(defclass OP4Env [ACE]
  """
  Base class for OP4
  """
  (defn __init__ [self &kwargs kwargs]
    (setv self.num-gmid 6
          self.num-fug 6
          self.num-ib 3)
    (.__init__ (super OP4Env self) #** (| kwargs {"ace_id" "op4"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. 
    """
    (let [unscaled-action (unscale-value action self.action-scale-min 
                                                self.action-scale-max)

          (, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
             gmid-ls1 gmid-ref)                   (as-> unscaled-action it
                                                      (get it (slice None 6)))
          (, fug-cm1  fug-cm2  fug-cm3  fug-dp1 
             fug-ls1  fug-ref)                    (as-> unscaled-action it
                                                      (get it (slice 6 12))
                                                      (np.power 10 it))
          (, i1 i2 i3)                              (as-> unscaled-action it
                                                      (get it (slice -3 None))
                                                      (np.array it)
                                                      (* it 1e-6))

          i0  (get self.design-constraints "i0"   "init")
          vdd (get self.design-constraints "vsup" "init")

          M1-lim (-> self (. design-constraints) (get "Mcm13" "max") (int))
          M2-lim (-> self (. design-constraints) (get "Mcm22" "max") (int))
          M3-lim (-> self (. design-constraints) (get "Mcm12" "max") (int))

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
          Wref (/ i3     (get ref-out 0)) ]

    ;(setv self.last-action (->> unscaled-action (zip self.input-parameters) (dict)))
    (setv self.last-action (dict (zip self.input-parameters
        [ gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 gmid-ls1 gmid-ref
          fug-cm1  fug-cm2  fug-cm3  fug-dp1 fug-ls1  fug-ref
          i1 i2 i3 ])))

    { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3 "Lc1" Lls1 "Lr" Lref
      "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3 "Wc1" Wls1 "Wr" Wref
      "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31"Mc1" Mls1 
                "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32  
                "Mcm13" Mcm13 
      #_/ })))

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

(defclass OP4XH035V2Env [OP4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V2Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 2}))))

(defclass OP4XH035V3Env [OP4Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V3Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 3}))))

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

(defclass OP4XH018V3Env [OP4Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH018V3Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 3}))))

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

(defclass OP4XT018V3Env [OP4Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XT018V3Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 3}))))

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

(defclass OP4SKY130V3Env [OP4Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4SKY130V3Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 3}))))

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

(defclass OP4GPDK180V3Env [OP4Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4GPDK180V3Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 3}))))
