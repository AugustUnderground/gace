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

(defclass OP6V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP6V0Env self) #** kwargs)

    ;; The action space consists of 14 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 14) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0 7.0 7.0       ; gm/Id min
                                           1e6 5e5 1e6 1e6 1e6 1e6       ; fug min
                                           3e-6 1.5e-6])                 ; branch currents
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0 17.0 17.0 ; gm/Id max
                                           1e9 5e8 1e9 1e9 1e9 1e9       ; fug max
                                           48e-6 480e-6]))               ; branch currents
    #_/ )

  (defn step ^(of tuple np.array float bool dict) [self ^np.array action]
    """
    Takes an array of electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    TODO: Implement sizing procedure.
    """
    (let [(, gmid-cm1 gmid-cm2 gmid-cs1 gmid-dp1 gmid-res gmid-cap
             fug-cm1  fug-cm2  fug-cs1  fug-dp1  fug-res  fug-cap
             i1 i2 ) (unscale-value action self.action-scale-min 
                                           self.action-scale-max)

          (, Mcm2 Mcs1 Mres Mcap Mdp1) (, 2 2 2 2 2)

          M1 (-> (/ self.i0 i1) (Fraction) (.limit-denominator 100))

          M2 (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))

          (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
          Mcm13 M2.denominator

          ;vx (/ self.vdd 2.7)

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2.0) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2.0) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2.0) 0.0]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (/ self.vdd 2.0) 0.0]])
          cap-in (np.array [[gmid-cap fug-cap              0.0  0.0]])
          res-in (np.array [[gmid-res fug-res              0.0  0.0]])
          
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cs1-out (first (self.pmos.predict cs1-in))
          dp1-out (first (self.nmos.predict dp1-in))
          cap-out (first (self.pmos.predict cap-in))
          res-out (first (self.pmos.predict res-in))

          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcs1 (get cs1-out 1)
          Ldp1 (get dp1-out 1)
          Lcap (get cap-out 1)
          Lres (get res-out 1)

          Wcm1 (/ self.i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wcs1 (/ i2 (get cs1-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 
          Wcap (/ i2 (get cap-out 0)) 
          Wres (/ i2 (get res-out 0)) 
          
          ;Vgs-cap (get cap-out 3)
          ;Vds-res (abs (- (/ self.vsup 2.0) Vgs-cap))
          ;Vbs-res (abs (- self.vsup 0.0))

          sizing {"Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs"   Lcs1  
                  "Ld"    Ldp1  "Lr1"   Lres  "Lc1"   Lcap
                  "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs"   Wcs1  
                  "Wd"    Wdp1  "Wr1"   Wres  "Wc1"   Wcap
                  "Mcm11" Mcm11 "Mcm21" Mcm2  "Mcs"   Mcs1  
                  "Md"    Mdp1  "Mr1"   Mres  "Mc1"   Mcap
                  "Mcm12" Mcm12 "Mcm13" Mcm13 "Mcm22" Mcm2
                  #_/ }]

    (self.size-circuit sizing))))

(defclass OP6V1Env [ACE]
  """
  Base class for electrical design space (v1)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP6V1Env self) #** kwargs)

    ;; The action space consists of 12 parameters ∈ [-1;1]. Ws and Ls for
    ;; each building block and mirror ratios as well as the cap and res.
    ;; [ "Wd" "Wcm1"  "Wcm2"  "Wcs" "Wc1" "Wr1"
    ;;   "Ld" "Lcm1"  "Lcm2"  "Lcs" "Lc1" "Lr1"
    ;;        "Mcm11"         "Mcs" "Mc1" "Mr1"
    ;;        "Mcm12"        
    ;;        "Mcm13"                           ]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 18) 
                                 :dtype np.float32)
          w-min (list (repeat 0.4e-6 6))  w-max (list (repeat 150e-6 6))
          l-min (list (repeat 0.35e-6 6)) l-max (list (repeat 15e-6 6))
          m-min [1 1 1 1 1 1]             m-max [3 40 4 4 10 40]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max)))
    #_/ )

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [ (, Wdp1 Wcm1  Wcm2  Wcs1 Wcap Wres 
              Ldp1 Lcm1  Lcm2  Lcs1 Lcap Lres 
                   Mcm11       Mcs1 Mcap Mres 
                   Mcm12        
                   Mcm13  ) (unscale-value action self.action-scale-min 
                                                  self.action-scale-max)
          
          (, Mdp1 Mcm2) (, 2 2)

          sizing {"Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs"   Lcs1  
                  "Ld"    Ldp1  "Lr1"   Lres  "Lc1"   Lcap
                  "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs"   Wcs1  
                  "Wd"    Wdp1  "Wr1"   Wres  "Wc1"   Wcap
                  "Mcm11" Mcm11 "Mcm21" Mcm2  "Mcs"   Mcs1  
                  "Md"    Mdp1  "Mr1"   Mres  "Mc1"   Mcap
                  "Mcm12" Mcm12 "Mcm13" Mcm13 "Mcm22" Mcm2
                  #_/ }]

      (self.size-circuit sizing))))

(defclass OP6XH035V0Env [OP6V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH035V0Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 235)}))))

(defclass OP6XH035V1Env [OP6V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6XH035V1Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "xh035-3V3" 
                          "ace_variant" 1 "obs_shape" (, 235)}))))
