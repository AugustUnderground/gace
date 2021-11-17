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
    ;; each building block and 2 branch currents.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 14) 
                                 :dtype np.float32)
          self.action-scale-min 
                (np.concatenate (, (np.repeat self.gmid-min 6)  ; gm/Id min
                                   (np.repeat self.fug-min  6)  ; fug min
                                   (np.array [(/ self.i0 3.0)   ; i1 = M11 : M12
                                              (/ self.i0 3.0)   ; i2 = M11 : M13
                                              #_/ ])))
          self.action-scale-max 
                (np.concatenate (, (np.repeat self.gmid-max 6)  ; gm/Id min
                                   (np.repeat self.fug-max  6)  ; fug min
                                   (np.array [(* self.i0 10.0)  ; i1 = M11 : M12
                                              (* self.i0 40.0)  ; i2 = M11 : M13
                                              #_/ ]))))

    ;; Specify Input Parameternames
    (setv self.input-parameters 
          [ "gmid-cm1" "gmid-cm2" "gmid-cs1" "gmid-dp1" "gmid-res" "gmid-cap"
            "fug-cm1"  "fug-cm2"  "fug-cs1"  "fug-dp1"  "fug-res"  "fug-cap"
            "i1" "i2" ]))

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

          i0 self.i0
          
          M1 (-> (/ i0 i1) (Fraction) (.limit-denominator 100))
          M2 (-> (/ i0 i2) (Fraction) (.limit-denominator 100))
          
          Mcm11 M1.numerator Mcm12 M1.denominator Mcm13 M2.denominator
          Mcm2 2
          Mcs1 2
          Mres 2
          Mcap 2
          Mdp1 2

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2.0) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2.0) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2.0) 0.0]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (/ self.vdd 2.0) 0.0]])
          cap-in (np.array [[gmid-cap fug-cap             0.0  0.0]])
          res-in (np.array [[gmid-res fug-res             0.0  0.0]])
          
          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cs1-out (first (self.pmos.predict cs1-in))
          cap-out (first (self.pmos.predict cap-in))
          res-out (first (self.pmos.predict res-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcs1 (get cs1-out 1)
          Lcap (get cap-out 1)
          Lres (get res-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i0     (get cm1-out 0))
          Wcm2 (/ i1 2.0 (get cm2-out 0))
          Wcs1 (/ i2     (get cs1-out 0))
          Wcap (/ i2     (get cap-out 0)) 
          Wres (/ i2     (get res-out 0)) 
          
          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2 "Lr1" Lres "Lc1" Lcap "Lcs" Lcs1  
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2 "Wcs" Wcs1 "Wr1" Wres "Wc1" Wcap
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm2 "Mcs" Mcs1 "Mr1" Mres "Mc1" Mcap
                             "Mcm12" Mcm12 "Mcm22" Mcm2
                             "Mcm13" Mcm13 
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
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 18) 
                                 :dtype np.float32)
          l-min (list (repeat self.l-min 6)) l-max (list (repeat self.l-max 6))
          w-min (list (repeat self.w-min 6)) w-max (list (repeat self.w-max 6))
          m-min [1 1 1 1 1 1]             m-max [3 40 4 4 10 40]
          self.action-scale-min (np.array (+ l-min w-min m-min))
          self.action-scale-max (np.array (+ l-max w-max m-max)))

    ;; Specify Input Parameternames
    (setv self.input-parameters [ "Ldp1" "Lcm1"  "Lcm2"  "Lcs1" "Lcap" "Lres"
                                  "Wdp1" "Wcm1"  "Wcm2"  "Wcs1" "Wcap" "Wres"
                                         "Mcm11"         "Mcs1" "Mcap" "Mres" 
                                         "Mcm12"
                                         "Mcm13" ]))

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [ (, Ldp1 Lcm1  Lcm2  Lcs1 Lcap Lres 
              Wdp1 Wcm1  Wcm2  Wcs1 Wcap Wres 
                   Mcm11       Mcs1 Mcap Mres 
                   Mcm12        
                   Mcm13  ) (unscale-value action self.action-scale-min 
                                                  self.action-scale-max)

          Mdp1 2
          Mcm2 2

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2 "Lr1" Lres "Lc1" Lcap "Lcs" Lcs1  
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2 "Wcs" Wcs1 "Wr1" Wres "Wc1" Wcap
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm2 "Mcs" Mcs1 "Mr1" Mres "Mc1" Mcap
                             "Mcm12" Mcm12 "Mcm22" Mcm2
                             "Mcm13" Mcm13 
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

(defclass OP6SKY130V0Env [OP6V0Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6SKY130V0Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "sky130-1V8" 
                          "ace_variant" 0 "obs_shape" (, 253)}))))

(defclass OP6SKY130V1Env [OP6V1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6SKY130V1Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "sky130-1V8" 
                          "ace_variant" 1 "obs_shape" (, 253)}))))

(defclass OP6GPDK180V0Env [OP6V0Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6GPDK180V0Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "gpdk180-1V8" 
                          "ace_variant" 0 "obs_shape" (, 305)}))))

(defclass OP6GPDK180V1Env [OP6V1Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP6GPDK180V1Env self) #**
               (| kwargs {"ace_id" "op6" "ace_backend" "gpdk180-1V8" 
                          "ace_variant" 1 "obs_shape" (, 305)}))))
