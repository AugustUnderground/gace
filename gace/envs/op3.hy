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

(defclass OP3V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; The action space consists of 10 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 11) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0        ; gm/Id min
                                           1e6 5e5 1e6 1e6        ; fug min
                                           3e-6 1.5e-6 1.5e-6])   ; branch currents
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0    ; gm/Id max
                                           1e9 5e8 1e9 1e9        ; fug max
                                           48e-6 480e-6 480e-6]))      ; branch currents

    ;; Parent constructor for initialization
    (.__init__ (super OP1V0Env self) #** kwargs))

  (defn step ^(of tuple np.array float bool dict) [self ^np.array action]
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

          (, Mcm31 Mcm32 Mdp1) (, 2 2 2)

          M1 (-> (/ self.i0 i1) (Fraction) (.limit-denominator 100))
          M2 (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))
          M3 (-> (/ (/ i1 2) i3) (Fraction) (.limit-denominator 100))

          (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
          (, Mcm2x1 Mcm212) (, M2.numerator M2.denominator)
          Mcm222  M3.denominator

          ;vx (/ self.vdd 2.7)

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vdd 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2) 0.0]])

          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cm3-out (first (self.nmos.predict cm3-in))
          dp1-out (first (self.nmos.predict dp1-in))

          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Ldp1 (get dp1-out 1)

          Wcm1 (/ self.i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wcm3 (/ i2 (get cm3-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 

          sizing { "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm2x1" Mcm2x1 "Mcm31" Mcm31 "Md" Mdp1 
                   "Mcm12" Mcm12 "Mcm212" Mcm212 "Mcm32" Mcm32 
                                 "Mcm222" Mcm222 
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP3V1Env [ACE]
  """
  Base class for geometric design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; The action space consists of 14 geometrical parameters ∈ [-1;1]:
    ;;  [ 'Wd', 'Wcm1',  'Wcm2',  'Wcm3'      # ∈ [ 0.4e-6  ; 150e-6 ]
    ;;  , 'Ld', 'Lcm1',  'Lcm2',  'Lcm3'      # ∈ [ 0.35e-6 ; 15e-6  ]
    ;;        , 'Mcm11', 'Mcm212', 'Mcm31'
    ;;        , 'Mcm12', 'Mcm222', 'Mcm32'
    ;;                 , 'Mcm2x1'
    ;;]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 15) 
                                 :dtype np.float32)
          w-min (list (repeat self.w-min 4)) w-max (list (repeat self.w-max 4))
          l-min (list (repeat self.l-min 4)) l-max (list (repeat self.l-max 4))
          m-min (list (repeat 1.0 7))     
          m-max [3 16 10 10 20 20 3] ;; M11, M12, M31, M32, M212, M222, M2x1
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max)))

    ;; Parent constructor for initialization
    (.__init__ (super OP3V1Env self) #** kwargs))

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wdp1 Wcm1  Wcm2   Wcm3 
             Ldp1 Lcm1  Lcm2   Lcm3  
             Mcm11 Mcm12 Mcm31 Mcm32
             Mcm212 Mcm222 Mcm2x1 ) (unscale-value action self.action-scale-min 
                                                          self.action-scale-max)

          Mdp1 2 

          sizing { "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm212" Mcm212 "Mcm31" Mcm31 "Md" Mdp1 
                   "Mcm12" Mcm12 "Mcm222" Mcm222 "Mcm32" Mcm32 
                                 "Mcm2x1" Mcm2x1 }]

      (self.size-circuit sizing))))

(defclass OP3XH035V0Env [OP3V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "xh035-3V3" 
                          "variant" 0 "obs_shape" (, 246)}))))

(defclass OP3XH035V1Env [OP3V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super NAND4XH035V1Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "xh035-3V3" 
                          "variant" 1 "obs_shape" (, 246)}))))
