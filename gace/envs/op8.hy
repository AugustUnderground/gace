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

(defclass OP8V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP8V0Env self) #** kwargs)

    ;; The action space consists of 14 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block, and 4 branch currents.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 16) 
                                 :dtype np.float32)
          self.action-scale-min 
                (np.concatenate (, (np.repeat self.gmid-min 6)                ; gm/Id min
                                   (np.repeat self.fug-min  6)                ; fug min
                                   (np.array [(/ self.i0 3.0)                 ; i1 = M51 : M53
                                              (/ self.i0 3.0)                 ; i2 = M51 : M52
                                              (/ (* (/ self.i0 3.0) 2.0) 3.0) ; i3 = M41 : M42
                                              (/ (* (/ self.i0 3.0) 2.0) 3.0) ; i4 = M41 : M43
                                              #_/ ])))
          self.action-scale-max 
                (np.concatenate (, (np.repeat self.gmid-max 6)                ; gm/Id min
                                   (np.repeat self.fug-max  6)                ; fug min
                                   (np.array [(* self.i0 15.0)                ; i1 = M51 : M53
                                              (* self.i0 3.0)                 ; i2 = M51 : M52
                                              (* self.i0 3.0 20.0)            ; i3 = M41 : M42
                                              (* self.i0 3.0 20.0)            ; i4 = M41 : M43
                                              #_/ ]))))

    ;; Specify Input Parameternames
    (setv self.input-parameters 
          [ "gmid-dp1" "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-cm4" "gmid-cm5" 
            "fug-dp1"  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-cm4"  "fug-cm5" 
            "i1" "i2" "i3" "i4" ]))

  (defn step ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-dp1 gmid-cm1 gmid-cm2 gmid-cm3 gmid-cm4 gmid-cm5
             fug-dp1  fug-cm1  fug-cm2  fug-cm3  fug-cm4  fug-cm5
             i1 i2 i3 i4 ) (unscale-value action self.action-scale-min 
                                                 self.action-scale-max)

          i0   self.i0

          M1 (-> (/ i0 i1) (Fraction) (.limit-denominator 100))
          M2 (-> (/ i0 i2) (Fraction) (.limit-denominator 100))
          M3 (-> (/ i2 i3) (Fraction) (.limit-denominator 100))
          M4 (-> (/ i2 i4) (Fraction) (.limit-denominator 100))

          Mdp1 2 Mcm1 2 Mcm2 2 Mcm3 2
          Mcm51 M1.numerator Mcm52 M2.denominator Mcm53 M1.denominator
          Mcm41 M3.numerator Mcm42 M3.denominator Mcm43 M4.denominator

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2.0) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2.0) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2.0) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vdd 2.0) 0.0]])
          cm4-in (np.array [[gmid-cm4 fug-cm4 (/ self.vdd 2.0) 0.0]])
          cm5-in (np.array [[gmid-cm5 fug-cm5 (/ self.vdd 2.0) 0.0]])

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

          Wdp1 (/ (/ i1 2.0) (get dp1-out 0)) 
          Wcm1 (/    i3      (get cm1-out 0))
          Wcm2 (/    i3      (get cm2-out 0))
          Wcm3 (/    i3      (get cm3-out 0))
          Wcm4 (/    i2      (get cm4-out 0))
          Wcm5 (/    i0      (get cm5-out 0))

          sizing { "Ld1" Ldp1 "Lcm1" Lcm1 "Lcm2" Lcm2 "Lcm3" Lcm3 "Lcm4"  Lcm4  "Lcm5"  Lcm5
                   "Wd1" Wdp1 "Wcm1" Wcm1 "Wcm2" Wcm2 "Wcm3" Wcm3 "Wcm4"  Wcm4  "Wcm5"  Wcm5
                   "Md1" Mdp1 "Mcm1" Mcm1 "Mcm2" Mcm2 "Mcm3" Mcm3 "Mcm41" Mcm41 "Mcm51" Mcm51
                                                                  "Mcm42" Mcm42 "Mcm52" Mcm52
                                                                  "Mcm43" Mcm42 "Mcm53" Mcm53 
                   #_/ }]
    (pp sizing)
    ;(self.size-circuit sizing)
    (self.size-circuit (ac.random-sizing self.ace))
    #_/ )))

(defclass OP8V1Env [ACE]
  """
  Base class for electrical design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP8V1Env self) #** kwargs)

    ;; The action space consists of 21 parameters ∈ [-1;1]. Ws and Ls for
    ;; each building block and mirror ratios as well as the cap and res.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 21) 
                                 :dtype np.float32)
          l-min (list (repeat self.l-min 6)) l-max (list (repeat self.l-max 6))
          w-min (list (repeat self.w-min 6)) w-max (list (repeat self.w-max 6))
          m-min [2  2  2  1 1 2  1 2  1]     m-max [20 20 20 3 3 20 3 20 15]
          self.action-scale-min (np.array (+ l-min w-min m-min))
          self.action-scale-max (np.array (+ l-max w-max m-max)))

    ;; Specify Input Parameternames
    (setv self.input-parameters [ "Ldp1" "Lcm1" "Lcm2" "Lcm3" "Lcm4"  "Lcm5"
                                  "Wdp1" "Wcm1" "Wcm2" "Wcm3" "Wcm4"  "Wcm5"
                                         "Mcm1" "Mcm2" "Mcm3" "Mcm41" "Mcm51" 
                                                              "Mcm42" "Mcm52" 
                                                              "Mcm43" "Mcm53" ]))

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [ (,  Ldp1 Lcm1 Lcm2 Lcm3 Lcm4  Lcm5 
               Wdp1 Wcm1 Wcm2 Wcm3 Wcm4  Wcm5 
                    Mcm1 Mcm2 Mcm3 Mcm41 Mcm51 
                                   Mcm42 Mcm52 
                                   Mcm43 Mcm53 ) (unscale-value action 
                                                                self.action-scale-min 
                                                                self.action-scale-max)
          
          Mdp1 2

          sizing { "Ld1" Ldp1 "Lcm1" Lcm1 "Lcm2" Lcm2 "Lcm3" Lcm3 "Lcm4"  Lcm4  "Lcm5"  Lcm5
                   "Wd1" Wdp1 "Wcm1" Wcm1 "Wcm2" Wcm2 "Wcm3" Wcm3 "Wcm4"  Wcm4  "Wcm5"  Wcm5
                   "Md1" Mdp1 "Mcm1" Mcm1 "Mcm2" Mcm2 "Mcm3" Mcm3 "Mcm41" Mcm41 "Mcm51" Mcm51
                                                                  "Mcm42" Mcm42 "Mcm52" Mcm52
                                                                  "Mcm43" Mcm42 "Mcm53" Mcm53 
                   #_/ } ]

      (self.size-circuit sizing))))

(defclass OP8XH035V0Env [OP8V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8XH035V0Env self) #**
               (| kwargs {"ace_id" "op8" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 277)}))))
  
(defclass OP8XH035V1Env [OP8V1Env]
"""
Implementation: xh035-3V3
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP8XH035V1Env self) #**
             (| kwargs {"ace_id" "op8" "ace_backend" "xh035-3V3" 
                        "ace_variant" 1 "obs_shape" (, 277)}))))

(defclass OP8GPDK180V0Env [OP8V0Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP8GPDK180V0Env self) #**
               (| kwargs {"ace_id" "op8" "ace_backend" "gpdk180-1V8" 
                          "ace_variant" 0 "obs_shape" (, 397)}))))
  
(defclass OP8GPDK180V1Env [OP8V1Env]
"""
Implementation: gpdk180-1V8
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP8GPDK180V1Env self) #**
             (| kwargs {"ace_id" "op8" "ace_backend" "gpdk180-1V8" 
                        "ace_variant" 1 "obs_shape" (, 397)}))))
