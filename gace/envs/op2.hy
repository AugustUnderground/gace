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

(import [.amp [SingleEndedOpAmpEnv]])
(import [.util.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass OP2Env [SingleEndedOpAmpEnv]
  """
  Derived amplifier class, implementing the Symmetrical Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]
    """
    Constructs a Symmetrical Amplifier Environment with XH035 device models and
    the corresponding netlist.
    Arguments:
      pdk-path:   This will be passed to the ACE backend.
      ckt-path:   This will be passed to the ACE backend.
      nmos-path:  Prefix path, expects to find `nmos-path/model.pt`, 
                  `nmos-path/scale.X` and `nmos-path/scale.Y` at this location.
      pmos-path:  Same as 'nmos-path', but for PMOS model.
      max-moves:  Maximum amount of steps the agent is allowed to take per
                  episode, before it counts as failed. Default = 200.
      random-target: Generate new random target for each episode.
      target: Specific target, if given, no random targets will be generated,
              and the agent tries to find the same one over and over again.

    """

    ;; ACE Environment ID
    (setv self.ace-env "op2")

    ;; Initialize parent Environment.
    (.__init__ (super OP2Env self) 
               [pdk-path] ckt-path
               nmos-path pmos-path
               max-moves
               :data-log-prefix data-log-prefix
               #_/ )

    ;; Generate random target of None was provided.
    (setv self.random-target random-target
          self.target        (or target 
                                 (self.target-specification :random random-target
                                                            :noisy True)))

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

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, 246)  :dtype np.float32)))

  (defn step [self action]
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

          ;vx (/ self.vsup 2.7)

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vsup 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vsup 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ self.vsup 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vsup 2) 0.0]])

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

      (.size-step (super) sizing)))

  (defn render [self &optional [mode "ascii"]]
    """
    Prints an ASCII Schematic of the Symmetrical Amplifier courtesy
    https://github.com/Blokkendoos/AACircuit
    """
    (cond [(= mode "ascii")
           (print f"
   #------------------o-------------o-------------o-------------.       
  VDD                 | MPCM222     |             |     MPCM212 |       
                      +-||       ||-+ MPCM221     +-||       ||-+       
                      <-||   Y   ||->             <-||   X   ||->       
                      +-||----o--||-+     MPCM211 +-||--o----||-+       
                      |       |     |             |     |       |       
                      |       |     |             |     |       |       
                      |       '-----o             o-----'       |       
                      |             |             |             |       
                      |             |             |             |       
       B              |             |             |             |       
       #              |          ||-+ MND11       +-||          |       
       |              |    INN   ||<-             ->||   INP    |       
       |              |     #----||-+       MND12 +-||----#     |       
       o------.       |             |     CM      |             |   OUT 
       |      |       |             '------o------'             o----#  
       |      |       |                    |                    |       
MNCM11 +-||   |       |                 ||-+ MNCM12             |       
       ->||   |       |                 ||<-                    |       
       +-||---o-------)-----------------||-+                    |       
       |              |                    |                    |       
       |              |                    |                    |       
       |              |                    |                    |       
       |              o------.             |                    |       
       |              |      |             |                    |       
       |              |      |             |                    |       
       |       MCNM31 +-||   |             |                 ||-+ MCNM32
       |              ->||   |  Z          |                 ||<-       
       |              +-||---o-------------)-----------------||-+       
  VSS  |              |                    |                    |       
   #---o--------------o--------------------o--------------------'       
" )]
          [True (.render (super) mode)])))

(defclass OP2XH035Env [OP2Env]
  """
  Symmetrical Amplifier in XH035 Technology.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]

    (.__init__ (super OP2XH035Env self) :pdk-path pdk-path :ckt-path ckt-path
                                        :nmos-path nmos-path :pmos-path pmos-path
                                        :max-moves max-moves :random-target random-target
                                        :target target :data-log-prefix data-log-prefix
                                        #_/ ))
 
  (defn target-specification ^dict [self &optional ^bool [random False] 
                                                   ^bool [noisy True]]
    """
    Generate a noisy target specification.
    """
    (let [ts {"a_0"         55.0
              "ugbw"        3750000.0
              "pm"          65.0
              "gm"          -30.0
              "sr_r"        3750000.0
              "sr_f"        -3750000.0
              "vn_1Hz"      5e-06
              "vn_10Hz"     2e-06
              "vn_100Hz"    5e-07
              "vn_1kHz"     1.5e-07
              "vn_10kHz"    5e-08
              "vn_100kHz"   2.5e-08
              "psrr_n"      80.0
              "psrr_p"      80.0
              "cmrr"        80.0
              "v_il"        0.9
              "v_ih"        3.2
              "v_ol"        1.65
              "v_oh"        3.2
              "i_out_min"   -7e-5
              "i_out_max"   7e-5
              "overshoot_r" 2.0
              "overshoot_f" 2.0
              "voff_stat"   3e-3
              "voff_sys"    -1.5e-3
              "A"           5.5e-10
              #_/ }
              factor (cond [random (np.abs (np.random.normal 1 0.5))]
                           [noisy  (np.random.normal 1 0.01)]
                           [True   1.0])]
      (dfor (, p v) (.items ts)
        [ p (if noisy (* v factor) v) ]))))

(defclass OP2XH035GeomEnv [OP2XH035Env]
  """
  Symmetrical Amplifier in XH035 Technology with geometrical action space.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]

    (.__init__ (super OP2XH035GeomEnv self) :pdk-path pdk-path 
                                            :ckt-path ckt-path
                                            :max-moves max-moves 
                                            :random-target random-target
                                            :target target 
                                            :data-log-prefix data-log-prefix
                                            #_/ )

    ;; The action space consists of 14 geometrical parameters ∈ [-1;1]:
    ;;  [ 'Wd', 'Wcm1',  'Wcm2',  'Wcm3'      # ∈ [ 0.4e-6  ; 150e-6 ]
    ;;  , 'Ld', 'Lcm1',  'Lcm2',  'Lcm3'      # ∈ [ 0.35e-6 ; 15e-6  ]
    ;;        , 'Mcm11', 'Mcm21'
    ;;        , 'Mcm12', 'Mcm22'        ]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 12) 
                                 :dtype np.float32)
          w-min (list (repeat 0.4e-6 4))  w-max (list (repeat 150e-6 4))
          l-min (list (repeat 0.35e-6 4)) l-max (list (repeat 15e-6 4))
          m-min (list (repeat 1.0 4))     m-max [3 3 16 20]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max))))

  (defn step [self action]
    """
    Takes an array of geometrical parameters for each building block.
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

      (.size-step (super) sizing))))


