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

(defclass OP1V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]
    ;; Parent constructor for initialization
    (.__init__ (super OP1V0Env self) #** kwargs)

    ;; The action space consists of 12 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block and 2 branch currents.
    (setv self.action-space (Box :low -1.0 :high 1.0 :shape (, 12) 
                                 :dtype np.float32)
          self.action-scale-min 
                (np.concatenate (, (np.repeat self.gmid-min 4)                    ; gm/Id min
                                   (np.repeat self.fug-min  4)                    ; fug min
                                   (np.array [self.Rc-min self.Cc-min])           ; Rc and Cc
                                   (np.array [(/ self.i0 3.0) (/ self.i0 3.0)]))) ; branch currents
          self.action-scale-max 
                (np.concatenate (, (np.repeat self.gmid-max 4)                    ; gm/Id min
                                   (np.repeat self.fug-max  4)                    ; fug min
                                   (np.array [self.Rc-max self.Cc-max])           ; Rc and Cc
                                   (np.array [(* self.i0 10) (* self.i0 40)]))))  ; branch currents
    #_/ )

  (defn step ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-cm1 gmid-cm2 gmid-cs1 gmid-dp1
             fug-cm1  fug-cm2  fug-cs1  fug-dp1 
             res cap i1 i2 ) (unscale-value action self.action-scale-min 
                                                   self.action-scale-max)
          
          Wres self.Wres 
          (, Mdp1 Mcm21 Mcm22) (, 2 2 2)
          (, Mcap Mcm11)       (, 1 1)

          Lres (* (/ res self.rs) Wres)
          Wcap (* (np.sqrt (/ cap self.cs)) self.Mcap)
          Mcm12 (* (/ self.i0 i1) Mcm11)
          Mcm13 (* (/ i1 i2) Mcm11)
          Mcs1 Mcm13

          ;vx 1.25 

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2) 0.0]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (/ self.vdd 2) 0.0]])

          cm1-out (first (self.nmos.predict cm1-in))
          dp1-out (first (self.nmos.predict dp1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cs1-out (first (self.pmos.predict cs1-in))

          Lcm1 (get cm1-out 1)
          Ldp1 (get dp1-out 1)
          Lcm2 (get cm2-out 1)
          Lcs1 (get cs1-out 1)

          Wcm1 (/ self.i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0))
          Wcs1 (/ i2 (get cs1-out 0))

          sizing {"Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs"   Lcs1  "Ld"    Ldp1 
                  "Lres"  Lres  "Mcm11" Mcm11 "Mcm12" Mcm12 "Mcm13" Mcm13 
                  "Mcm21" Mcm21 "Mcm22" Mcm22 "Mcs"   Mcs1  "Md"    Mdp1  
                  "Mcap"  Mcap  "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs"   Wcs1 
                  "Wd"    Wdp1  "Wres"  Wres  "Wcap"  Wcap} ]

    (self.size-circuit sizing))))

(defclass OP1V1Env [ACE]
  """
  Base class for geometric design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP1V1Env self) #** kwargs)

    ;; The action space consists of 14 parameters ∈ [-1;1]. Ws and Ls for
    ;; each building block and mirror ratios as well as the cap and res.
    ;; [ "Wd" "Wcm1"  "Wcm2"  "Wcs" "Wcap"  "Wres"
    ;;   "Ld" "Lcm1"  "Lcm2"  "Lcs"         "Lres"
    ;;        "Mcm11"         "Mcs"
    ;;        "Mcm12" 
    ;;        "Mcm13"                             ]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 15) 
                                 :dtype np.float32)

          w-min (list (repeat self.w-min 6)) w-max (list (repeat self.w-max 6))
          l-min (list (repeat self.l-min 5)) l-max (list (repeat self.l-max 5))
          m-min [1  1  1  1 ]                m-max [3  40 10 40]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max)))
    #_/ )

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [ (, Wdp1 Wcm1 Wcm2 Wcs1 Wcap Wres
              Ldp1 Lcm1 Lcm2 Lcs1      Lres 
                   Mcm11     Mcs1 
                   Mcm12 
                   Mcm13 ) (unscale-value action self.action-scale-min 
                                                 self.action-scale-max)
          
          (, Mdp1 Mcm21 Mcm22) (, 2 2 2)
          Mcap 1 

          sizing {"Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs"   Lcs1  "Ld"    Ldp1 
                  "Lres"  Lres  "Mcm11" Mcm11 "Mcm12" Mcm12 "Mcm13" Mcm13 
                  "Mcm21" Mcm21 "Mcm22" Mcm22 "Mcs"   Mcs1  "Md"    Mdp1  
                  "Mcap"  Mcap  "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs"   Wcs1 
                  "Wd"    Wdp1  "Wres"  Wres  "Wcap"  Wcap}]

      (self.size-circuit sizing))))

(defclass OP1XH035V0Env [OP1V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1XH035V0Env self) #**
               (| kwargs {"ace_id" "op1" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 211)}))))
  
(defclass OP1XH035V1Env [OP1V1Env]
"""
Implementation: xh035-3V3
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1XH035V1Env self) #**
             (| kwargs {"ace_id" "op1" "ace_backend" "xh035-3V3" 
                        "ace_variant" 1 "obs_shape" (, 211)}))))

(defclass OP1GPDK180V0Env [OP1V0Env]
  """
  Implementation: gpdk180-1V2
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP1GPDK180V0Env self) #**
               (| kwargs {"ace_id" "op1" "ace_backend" "gpdk180-1V2" 
                          "ace_variant" 0 "obs_shape" (, 211)}))))
  
(defclass OP1GPDK180V1Env [OP1V1Env]
"""
Implementation: gpdk180-1V2
"""
(defn __init__ [self &kwargs kwargs]
  (.__init__ (super OP1GPDK180V1Env self) #**
             (| kwargs {"ace_id" "op1" "ace_backend" "gpdk180-1V2" 
                        "ace_variant" 1 "obs_shape" (, 211)}))))
