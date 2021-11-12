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

(defclass OP5V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP5V0Env self) #** kwargs)

    ;; The action space consists of 16 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 16) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0 7.0 7.0        ; gm/Id min
                                           1e6 5e5 1e6 1e6 1e6 1e6        ; fug min
                                           3e-6 1.5e-6 1.5e-6 1.5e-6])    ; branch currents
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0 17.0 17.0  ; gm/Id max
                                           1e9 5e8 1e9 1e9 1e9 1e9        ; fug max
                                           48e-6 480e-6 480e-6 480e-6]))  ; branch currents
    #_/ )

  (defn step ^(of tuple np.array float bool dict) [self ^np.array action]
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

          Mdp1 2

          M4 (-> (/ i2 i4) (Fraction) (.limit-denominator 100))
          (, Mls11 Mls12) (, M4.numerator M4.denominator)
          (, Mcm31 Mcm32) (, M4.numerator M4.denominator)

          M1 (-> (/ self.i0 i1) (Fraction) (.limit-denominator 100))
          M21 (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))
          M22 (-> (/ (/ i1 2) i4) (Fraction) (.limit-denominator 100))

          (, Mcm11 Mcm13) (, M1.numerator M1.denominator)
          Mcm12 (* (/ i1 self.i0) Mcm11)

          (, Mcm2x1 Mcm212) (, M21.numerator M21.denominator)
          Mcm222 M22.denominator

          ;vx (/ self.vdd 2.7)

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vdd 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2) 0.0]])
          ls1-in (np.array [[gmid-ls1 fug-ls1 (/ self.vdd 2) 0.0]])
          ref-in (np.array [[gmid-ref fug-ref (/ self.vdd 2) 0.0]])

          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cm3-out (first (self.nmos.predict cm3-in))
          dp1-out (first (self.nmos.predict dp1-in))
          ls1-out (first (self.pmos.predict ls1-in))
          ref-out (first (self.pmos.predict ref-in))

          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Ldp1 (get dp1-out 1)
          Lls1 (get ls1-out 1)
          Lref (get ref-out 1)

          Wcm1 (/ self.i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wcm3 (/ i2 (get cm3-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 
          Wls1 (/ i2 (get ls1-out 0)) 
          Wref (/ i3 (get ref-out 0)) 

          sizing { "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Ld" Ldp1 "Lc1"  Lls1  "Lr" Lref
                   "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wd" Wdp1 "Wc1"  Wls1  "Wr" Wref
                   "Mcm11" Mcm11 "Mcm212" Mcm212 "Mcm31" Mcm31 "Md" Mdp1 "Mc11" Mls11 
                   "Mcm12" Mcm12 "Mcm222" Mcm222 "Mcm32" Mcm32           "Mc12" Mls12
                   "Mcm13" Mcm13 "Mcm2x1" Mcm2x1
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP5V1Env [ACE]
  """
  Base class for electrical design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP5V1Env self) #** kwargs)

    ;; The action space consists of 22 parameters ∈ [-1;1]. 
    ;; [ "Wd" "Wcm1"  "Wcm2"   "Wcm3"  "Wc1"  "Wr"
    ;;   "Ld" "Lcm1"  "Lcm2"   "Lcm3"  "Lc1"  "Lr"
    ;;        "Mcm11" "Mcm212" "Mcm31" "Mc11"
    ;;        "Mcm12" "Mcm222" "Mcm32" "Mc12"
    ;;        "Mcm13" "Mcm2x1                    ]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 22) 
                                 :dtype np.float32)
          w-min (list (repeat self.w-min 6)) w-max (list (repeat self.w-max 6))
          l-min (list (repeat self.l-min 6)) l-max (list (repeat self.l-max 6))
          m-min [1 1 1 1 1 1 1 1 1 1]        m-max [3 20 10 20 3 20 10 20 16 3]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max)))
    #_/ )

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wdp1 Wcm1  Wcm2   Wcm3  Wls1  Wref 
             Ldp1 Lcm1  Lcm2   Lcm3  Lls1  Lref 
                  Mcm11 Mcm212 Mcm31 Mls11 
                  Mcm12 Mcm222 Mcm32 Mls12
                  Mcm13 Mcm2x1) (unscale-value action self.action-scale-min 
                                               self.action-scale-max)

          Mdp1 2

          sizing { "Lcm1"  Lcm1  "Lcm2"   Lcm2   "Lcm3"  Lcm3  "Ld" Ldp1 "Lc1"  Lls1  "Lr" Lref
                   "Wcm1"  Wcm1  "Wcm2"   Wcm2   "Wcm3"  Wcm3  "Wd" Wdp1 "Wc1"  Wls1  "Wr" Wref
                   "Mcm11" Mcm11 "Mcm212" Mcm212 "Mcm31" Mcm31 "Md" Mdp1 "Mc11" Mls11
                   "Mcm12" Mcm12 "Mcm222" Mcm222 "Mcm32" Mcm32           "Mc12" Mls12
                   "Mcm13" Mcm13 "Mcm2x1" Mcm2x1
                  #_/ }]

      (self.size-circuit sizing))))

(defclass OP5XH035V0Env [OP5V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH035V0Env self) #**
               (| kwargs {"ace_id" "op5" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 285)}))))

(defclass OP5XH035V1Env [OP5V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP5XH035V1Env self) #**
               (| kwargs {"ace_id" "op5" "ace_backend" "xh035-3V3" 
                          "ace_variant" 1 "obs_shape" (, 285)}))))
