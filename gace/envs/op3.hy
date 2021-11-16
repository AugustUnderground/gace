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

    ;; Parent constructor for initialization
    (.__init__ (super OP3V0Env self) #** kwargs)

    ;; The action space consists of 11 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block and 3 branch currents.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 11) 
                                 :dtype np.float32)
          self.action-scale-min 
                (np.concatenate (, (np.repeat self.gmid-min 4)        ; gm/Id min
                                   (np.repeat self.fug-min  4)        ; fug min
                                   (np.array [(/ self.i0 3.0)         ; i1 = M11 : M12
                                              (/ self.i0 3.0 2.0 3.0) ; i2 = M211 : M212
                                              (/ self.i0 3.0 2.0 3.0) ; i3 = M221 : M222
                                              #_/ ])))
          self.action-scale-max 
                (np.concatenate (, (np.repeat self.gmid-max 4)     ; gm/Id min
                                   (np.repeat self.fug-max  4)     ; fug min
                                   (np.array [(* self.i0 16.0)     ; i1 = M11 : M12
                                              (* self.i0 8.0 20.0) ; i2 = M211 : M212
                                              (* self.i0 8.0 20.0) ; i3 = M221 : M222
                                              #_/ ]))))
    #_/ )

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

          i0 self.i0

          M1 (->    (/ i0 i1)      (Fraction) (.limit-denominator 100))
          M2 (-> (/ (/ i1 2.0) i2) (Fraction) (.limit-denominator 100))
          M3 (-> (/ (/ i1 2.0) i3) (Fraction) (.limit-denominator 100))
          M4 (->    (/ i2 i3)      (Fraction) (.limit-denominator 100))

          Mcm11  M1.numerator Mcm12 M1.denominator
          Mcm31  M4.numerator Mcm32 M4.denominator
          Mcm2x1 2 Mcm212 (round (/ i1 i3)) Mcm222 (round (/ i1 i2))
          Mdp1   2

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vdd 2) 0.0]])

          dp1-out (first (self.nmos.predict dp1-in))
          cm1-out (first (self.nmos.predict cm1-in))
          cm2-out (first (self.pmos.predict cm2-in))
          cm3-out (first (self.nmos.predict cm3-in))

          Ldp1 (get dp1-out 1)
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)

          Wdp1 (/ i1 2.0 (get dp1-out 0)) 
          Wcm1 (/ i0     (get cm1-out 0))
          Wcm2 (/ i1 2.0 (get cm2-out 0))
          Wcm3 (/ i2     (get cm3-out 0))

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

    ;; Parent constructor for initialization
    (.__init__ (super OP3V1Env self) #** kwargs)

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
    #_/ )

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
    (.__init__ (super OP3XH035V0Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 246)}))))

(defclass OP3XH035V1Env [OP3V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3XH035V1Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "xh035-3V3" 
                          "ace_variant" 1 "obs_shape" (, 246)}))))

(defclass OP3SKY130V0Env [OP3V0Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3SKY130V0Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "sky130-1V8" 
                          "ace_variant" 0 "obs_shape" (, 266)}))))

(defclass OP3SKY130V1Env [OP3V1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3SKY130V1Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "sky130-1V8" 
                          "ace_variant" 1 "obs_shape" (, 266)}))))

(defclass OP3GPDK180V0Env [OP3V0Env]
  """
  Implementation: gpdk180-1V2
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3GPDK180V0Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "gpdk180-1V2" 
                          "ace_variant" 0 "obs_shape" (, 294)}))))

(defclass OP3GPDK180V1Env [OP3V1Env]
  """
  Implementation: gpdk180-1V2
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP3GPDK180V1Env self) #**
               (| kwargs {"ace_id" "op3" "ace_backend" "gpdk180-1V2" 
                          "ace_variant" 1 "obs_shape" (, 294)}))))
