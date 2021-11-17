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

(defclass OP4V0Env [ACE]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP4V0Env self) #** kwargs)

    ;; The action space consists of 15 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block and 3 branch currents.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 15) 
                                 :dtype np.float32)
          self.action-scale-min 
                (np.concatenate (, (np.repeat self.gmid-min 6)      ; gm/Id min
                                   (np.repeat self.fug-min  6)      ; fug min
                                   (np.array [(/ self.i0 3.0)       ; i1 = M11 : M13
                                              (/ self.i0 2.0 3.0)   ; i2 = M21 : M22
                                              (/ self.i0 3.0)       ; i3 = M11 : M12
                                              #_/ ])))
          self.action-scale-max 
                (np.concatenate (, (np.repeat self.gmid-max 6)      ; gm/Id min
                                   (np.repeat self.fug-max  6)      ; fug min
                                   (np.array [(* self.i0 16.0)      ; i1 = M11 : M13
                                              (* self.i0 8.0 20.0)  ; i2 = M21 : M22
                                              (* self.i0 3.0)       ; i3 = M11 : M12
                                              #_/ ]))))

    ;; Specify Input Parameternames
    (setv self.input-parameters [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1" "gmid-ls1" "gmid-ref" 
                                  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"  "fug-ls1"  "fug-ref" 
                                  "i1" "i2" "i3" ]))

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
             i1 i2 i3 ) (unscale-value action self.action-scale-min 
                                              self.action-scale-max)

          i0 self.i0


          M1 (-> (/ i0 i1)     (Fraction) (.limit-denominator 100))
          M2 (-> (/ i1 2.0 i2) (Fraction) (.limit-denominator 100))
          M3 (-> (/ i0 i3)     (Fraction) (.limit-denominator 100))

          Mdp1  2
          Mcm11 M1.numerator Mcm12 M3.denominator Mcm13 M1.denominator
          Mcm21 M2.numerator Mcm22 M2.denominator
          Mcm31 2            Mcm31 2
          Mls1  Mcm22

          ;vx (/ self.vdd 2.7)

          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vdd 2) 0.0]])
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vdd 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vdd 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vdd 2) 0.0]])
          ls1-in (np.array [[gmid-ls1 fug-ls1 (/ self.vdd 2) 0.0]])
          ref-in (np.array [[gmid-ref fug-ref (/ self.vdd 2) 0.0]])

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

(defclass OP4V1Env [ACE]
  """
  Base class for geometric design space (v0)
  """
  (defn __init__ [self &kwargs kwargs]

    ;; Parent constructor for initialization
    (.__init__ (super OP4V1Env self) #** kwargs)

    ;; The action space consists of 18 parameters ∈ [-1;1]. 
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 18) 
                                 :dtype np.float32)
          l-min (list (repeat self.l-min 6)) l-max (list (repeat self.l-max 6))
          w-min (list (repeat self.w-min 6)) w-max (list (repeat self.w-max 6))
          m-min [1 1 1 1 1 1]                m-max [3 3 20 3 20 16]
          self.action-scale-min (np.array (+ l-min w-min m-min))
          self.action-scale-max (np.array (+ l-max w-max m-max)))

    ;; Specify Input Parameternames
    (setv self.input-parameters [ "Ldp1" "Lcm1"  "Lcm2"  "Lcm3"  "Lls1" "Lref" 
                                  "Wdp1" "Wcm1"  "Wcm2"  "Wcm3"  "Wls1" "Wref"
                                         "Mcm11" "Mcm21"         "Mls1" 
                                         "Mcm12" "Mcm22" 
                                         "Mcm13" ]))

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Ldp1 Lcm1  Lcm2  Lcm3  Lls1 Lref 
             Wdp1 Wcm1  Wcm2  Wcm3  Wls1 Wref 
                  Mcm11 Mcm21       Mls1 
                  Mcm12 Mcm22 
                  Mcm13 )           (unscale-value action self.action-scale-min 
                                                          self.action-scale-max)

          Mcm31 2 Mcm32 2 Mdp1 2

          sizing { "Ld" Ldp1 "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3 "Lc1" Lls1 "Lr" Lref
                   "Wd" Wdp1 "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3 "Wc1" Wls1 "Wr" Wref
                   "Md" Mdp1 "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31"Mc1" Mls1 
                             "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32  
                             "Mcm13" Mcm13 
                  #_/ }]

      (self.size-circuit sizing))))

(defclass OP4XH035V0Env [OP4V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V0Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "xh035-3V3" 
                          "ace_variant" 0 "obs_shape" (, 285)}))))

(defclass OP4XH035V1Env [OP4V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4XH035V1Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "xh035-3V3" 
                          "ace_variant" 1 "obs_shape" (, 285)}))))

(defclass OP4SKY130V0Env [OP4V0Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4SKY130V0Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "sky130-1V8" 
                          "ace_variant" 0 "obs_shape" (, 305)}))))

(defclass OP4SKY130V1Env [OP4V1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4SKY130V1Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "sky130-1V8" 
                          "ace_variant" 1 "obs_shape" (, 305)}))))

(defclass OP4GPDK180V0Env [OP4V0Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4GPDK180V0Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "gpdk180-1V8" 
                          "ace_variant" 0 "obs_shape" (, 383)}))))

(defclass OP4GPDK180V1Env [OP4V1Env]
  """
  Implementation: gpdk180-1V8
  """
  (defn __init__ [self &kwargs kwargs]
    (.__init__ (super OP4GPDK180V1Env self) #**
               (| kwargs {"ace_id" "op4" "ace_backend" "gpdk180-1V8" 
                          "ace_variant" 1 "obs_shape" (, 383)}))))
