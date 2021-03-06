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

(defclass OP1Env [ACE]
  """
  Base class for OP1
  """
  (defn __init__ [self &kwargs kwargs]
    (setv self.num-gmid 4
          self.num-fug 4
          self.num-ib 2)
    (.__init__ (super OP1Env self) #** (| kwargs {"ace_id" "op1"})))

  (defn step-v0 ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. 
    """
    (let [unscaled-action (unscale-value action self.action-scale-min 
                                                self.action-scale-max)

          (, gmid-cm1 gmid-cm2 gmid-cs1 gmid-dp1) (as-> unscaled-action it
                                                      (get it (slice None 4)))
          (, fug-cm1  fug-cm2  fug-cs1  fug-dp1 ) (as-> unscaled-action it
                                                      (get it (slice 4 8))
                                                      (np.power 10 it))
          ;res                                     (as-> unscaled-action it
          ;                                            (get it 8)
          ;                                            (* it 1e3))
          ;                                            ;(np.power 10 it)
          ;cap                                     (as-> unscaled-action it
          ;                                            (get it 9)
          ;                                            (* it 1e-12))
          (, i1 i2 )                              (as-> unscaled-action it
                                                      (get it (slice -2 None))
                                                      (* it 1e-6))

          ; sr   (* (get self.target "sr_r") 0.99)
          ; cl   (get self.design-constraints "cl" "init")
          ; cap  (/ i1 sr)
          ; res  (* (/ 1 (* gmid-cs1 i2)) (/ (+ cap cl) cap))

          i0   (get self.design-constraints "i0"   "init")
          vdd  (get self.design-constraints "vsup" "init")
          Wres (get self.design-constraints "Wres" "init")
          Mcap (get self.design-constraints "Mcap" "init")

          M1-lim (-> self (. design-constraints) (get "Mcm12" "max") (int))
          M2-lim (-> self (. design-constraints) (get "Mcm13" "max") (int))
          
          M1 (-> (/ i0 i1) (Fraction) (.limit-denominator M1-lim))
          M2 (-> (/ i0 i2) (Fraction) (.limit-denominator M2-lim))

          Mcm11 M1.numerator Mcm12 M1.denominator 
          Mcm13 (// (* M1.numerator M2.denominator) M2.numerator)
          Mcm21 (get self.design-constraints "Mcm21" "init")            
          Mcm22 (get self.design-constraints "Mcm22" "init")
          Mdp1  (get self.design-constraints "Md"    "init")            

          ;Lres (* (/ res rc) Wres)
          ;Wcap (* (np.sqrt (/ cap cc)) Mcap)
          ;Lres (res2len self.ace-backend res)
          ;Wcap (cap2wid self.ace-backend cap)
          Lres (get self.design-constraints "Lres" "init")
          Wcap (get self.design-constraints "Wcap" "init")

          cm1-in (np.array [[gmid-cm1 fug-cm1    (/ vdd 4.0)               0.0  ]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (- (/ vdd 3.0))              0.0  ]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (- (/ vdd 2.0))              0.0  ]])
          dp1-in (np.array [[gmid-dp1 fug-dp1    (/ vdd 3.0)     (- (/ vdd 4.0))]])

          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cs1-out (first (self.pmos.predict cs1-in))
          dp1-out (first (self.nmos.predict dp1-in))

          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcs1 (get cs1-out 1)
          Ldp1 (get dp1-out 1)

          Wcm1 (/ i0     (get cm1-out 0) Mcm11)
          Wcm2 (/ i1 2.0 (get cm2-out 0) Mcm21)
          Wdp1 (/ i1 2.0 (get dp1-out 0) Mdp1)
          Wcs  (/ i2     (get cs1-out 0)) 

          Mcs1 (-> Wcs (/ (get self.design-constraints "Wcs" "max")) (np.ceil) (.item) (int))
          Wcs1 (/ Wcs Mcs1)

          #_/ ]

    ;(setv self.last-action (->> unscaled-action (zip self.input-parameters) (dict)))
    (setv self.last-action (dict (zip self.input-parameters
        [ gmid-cm1 gmid-cm2 gmid-cs1 gmid-dp1
          fug-cm1  fug-cm2  fug-cs1  fug-dp1
          ;res cap 
          i1 i2 ])))

    { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs" Lcs1 "Lres" Lres  
      "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs" Wcs1 "Wres" Wres "Wcap" Wcap
      "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcs" Mcs1             "Mcap" Mcap  
                "Mcm12" Mcm12 "Mcm22" Mcm22
                "Mcm13" Mcm13 
      #_/ })))

(defclass OP1XH035V0Env [OP1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1XH035V0Env self) #**
               (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 0}))))
  
(defclass OP1XH035V1Env [OP1Env]
"""
Implementation: xh035-3V3
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XH035V1Env self) #**
             (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 1}))))

(defclass OP1XH035V3Env [OP1Env]
"""
Implementation: xh035-3V3
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XH035V3Env self) #**
             (| kwargs {"ace_backend" "xh035-3V3" "ace_variant" 3}))))

(defclass OP1XH018V0Env [OP1Env]
  """
  Implementation: xh018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1XH018V0Env self) #**
               (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 0}))))
  
(defclass OP1XH018V1Env [OP1Env]
"""
Implementation: xh018-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XH018V1Env self) #**
             (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 1}))))

(defclass OP1XH018V3Env [OP1Env]
"""
Implementation: xh018-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XH018V3Env self) #**
             (| kwargs {"ace_backend" "xh018-1V8" "ace_variant" 3}))))

(defclass OP1XT018V0Env [OP1Env]
  """
  Implementation: xt018-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1XT018V0Env self) #**
               (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 0}))))
  
(defclass OP1XT018V1Env [OP1Env]
"""
Implementation: xt018-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XT018V1Env self) #**
             (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 1}))))

(defclass OP1XT018V3Env [OP1Env]
"""
Implementation: xt018-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XT018V3Env self) #**
             (| kwargs {"ace_backend" "xt018-1V8" "ace_variant" 3}))))

(defclass OP1SKY130V0Env [OP1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1SKY130V0Env self) #**
               (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 0}))))
  
(defclass OP1SKY130V1Env [OP1Env]
"""
Implementation: sky130-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1SKY130V1Env self) #**
             (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 1}))))

(defclass OP1SKY130V3Env [OP1Env]
"""
Implementation: sky130-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1SKY130V3Env self) #**
             (| kwargs {"ace_backend" "sky130-1V8" "ace_variant" 3}))))
(defclass OP1GPDK180V0Env [OP1Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1GPDK180V0Env self) #**
               (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 0}))))
  
(defclass OP1GPDK180V1Env [OP1Env]
"""
Implementation: gpdk180-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1GPDK180V1Env self) #**
             (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 1}))))

(defclass OP1GPDK180V3Env [OP1Env]
"""
Implementation: gpdk180-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1GPDK180V3Env self) #**
             (| kwargs {"ace_backend" "gpdk180-1V8" "ace_variant" 3}))))
