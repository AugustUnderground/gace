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
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

;; Symmetrical Amplifier (OP2) Base Class

(defclass OP2Env [ACE]
  """
  Base class for symmetrical amplifier (op2)
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] 
                                 ^int [obs-shape 0]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; ACE ID, required by parent
    (setv self.ace-id "op2")

    ;; Call Parent Contructor
    (.__init__ (super OP2Env self) max-steps target random-target noisy-target 
                                        data-log-path param-log-path)

    ;; ACE setup
    (setv self.ace-constructor (ace-constructor self.ace-id self.ace-backend 
                                                :ckt ckt-path :pdk [pdk-path])
          self.ace (self.ace-constructor))

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, obs-shape)  :dtype np.float32))))

(defclass OP2V0Env [OP2Env]
  """
  Base class for electrical design space (v0)
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None]
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 206]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; Parent constructor for initialization
    (.__init__ (super OP2V0Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs-shape obs-shape
               :data-log-path data-log-path :param-log-path param-log-path)

        ;; Primitive Device setup
    (setv self.nmos (load-primitive "nmos" self.ace-backend :dev-path nmos-path)
          self.pmos (load-primitive "pmos" self.ace-backend :dev-path pmos-path))

    ;; The action space consists of 10 parameters ∈ [-1;1]. One gm/id and fug
    ;; for ;; each building block. This is subject to change and will include
    ;; branch currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 10) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0      ; gm/Id min
                                           1e6 5e5 1e6 1e6      ; fug min
                                           3e-6 1.5e-6])        ; branch currents
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0  ; gm/Id max
                                           1e9 5e8 1e9 1e9      ; fug max
                                           48e-6 480e-6]))      ; branch currents
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
             i1 i2 ) (unscale-value action self.action-scale-min 
                                           self.action-scale-max)

          (, Mcm31 Mcm32 Mdp1) (, 2 2 2)

          M1 (-> (/ self.i0 i1) (Fraction) (.limit-denominator 100))
          M2 (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))

          (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
          (, Mcm21 Mcm22) (, M2.numerator M2.denominator)

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

          sizing { "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31 "Md" Mdp1 
                   "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32 }]

    (self.size-circuit sizing))))

(defclass OP2V1Env [OP2Env]
  """
  Base class for geometric design space (v1)
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 206]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; Parent constructor for initialization
    (.__init__ (super OP2V1Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs-shape obs-shape
               :data-log-path data-log-path :param-log-path param-log-path)

    ;; The action space consists of 14 parameters ∈ [-1;1]. Ws and Ls for
    ;; each building block and mirror ratios as well as the cap and res.
    ;; [ "Wd" "Wcm1"  "Wcm2"  "Wcs" "Wcap"  "Wres"
    ;;   "Ld" "Lcm1"  "Lcm2"  "Lcs"         "Lres"
    ;;        "Mcm11"         "Mcs"
    ;;        "Mcm12" 
    ;;        "Mcm13"                             ]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 12) 
                                 :dtype np.float32)

          w-min (list (repeat self.w-min 4)) w-max (list (repeat self.w-max 4))
          l-min (list (repeat self.l-min 4)) l-max (list (repeat self.l-max 4))
          m-min (list (repeat 1.0 4))        m-max [3 3 16 20]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max)))
    #_/ )

  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wdp1 Wcm1  Wcm2 Wcm3 
             Ldp1 Lcm1  Lcm2 Lcm3  
                  Mcm11 Mcm21  
                  Mcm12 Mcm22 ) (unscale-value action self.action-scale-min 
                                                      self.action-scale-max)

          (, Mcm31 Mcm32 Mdp1) (, 2 2 2)

          sizing { "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31 "Md" Mdp1 
                   "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32 }]

      (self.size-circuit sizing))))

;; Technology Specific Implementations

(defclass OP2XH035V0Env [OP2V0Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None]
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 206]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    (setv self.ace-backend "xh035-3V3")

    (for [(, k v) (-> self.ace-backend (technology-data) (.items))]
      (setattr self k v))

    (.__init__ (super OP2XH035V0Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :nmos-path nmos-path :pmos-path pmos-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs_shape 206
               :data-log-path data-log-path :param-log-path param-log-path)
    #_/ ))

(defclass OP2XH035V1Env [OP2V1Env]
  """
  Implementation: xh035-3V3
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 206]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    (setv self.ace-backend "xh035-3V3")

    (for [(, k v) (-> self.ace-backend (technology-data) (.items))]
      (setattr self k v))

    (.__init__ (super OP2XH035V1Env self)
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs-shape obs-shape
               :data-log-path data-log-path :param-log-path param-log-path)))



(defclass OP2SKY130V0Env [OP2V0Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 266]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    (setv self.ace-backend "sky130-1V8")

    (for [(, k v) (-> self.ace-backend (technology-data) (.items))]
      (setattr self k v))

    (.__init__ (super OP2SKY130V1Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs-shape obs-shape
               :data-log-path data-log-path :param-log-path param-log-path)
    #_/ ))

(defclass OP2SKY130V1Env [OP2V1Env]
  """
  Implementation: sky130-1V8
  """
  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^bool [random-target False] ^bool [noisy-target True]
                                 ^dict [target None] ^int [max-steps 200] ^int [obs-shape 266]
                                 ^str [data-log-path ""] ^str [param-log-path "."]]

    (setv self.ace-backend "sky130-1V8")

    (for [(, k v) (-> self.ace-backend (technology-data) (.items))]
      (setattr self k v))

    (.__init__ (super OP2SKY130V1Env self) 
               :pdk-path pdk-path :ckt-path ckt-path
               :random-target random-target :noisy-target noisy-target
               :max-steps max-steps :obs-shape obs-shape
               :data-log-path data-log-path :param-log-path param-log-path)
    #_/ ))
