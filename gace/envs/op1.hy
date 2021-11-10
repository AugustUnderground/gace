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

(defclass OP1Env [SingleEndedOpAmpEnv]
  """
  Derived amplifier class, implementing the Miller Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]
    """
    Constructs a Miller Amplifier Environment with XH035 device models and
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
    (setv self.ace-env "op1")

    ;; Initialize parent Environment.
    (.__init__ (super OP1Env self) 
               [pdk-path] ckt-path
               nmos-path pmos-path
               max-moves random-target
               :data-log-prefix data-log-prefix
               #_/ )

    ;; Generate random target of None was provided.
    (setv self.random-target random-target
          self.target        (or target 
                                 (self.target-specification :random random-target
                                                            :noisy True)))

    ;; The action space consists of 12 parameters ∈ [-1;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 12) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0      ; gm/Id min
                                           1e6 1e6 1e6 1e6      ; fug min
                                           0.5e3 0.5e-12        ; Rc and Cc
                                           3e-6 1.5e-6])        ; branch currents
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0  ; gm/Id max
                                           1e9 5e8 1e9 1e9      ; fug max
                                           50e3 5e-12           ; Rc and Cc
                                           48e-6 480e-6]))      ; branch currents

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, 150)  :dtype np.float32)))

  (defn step [self action]
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

          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ self.vsup 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ self.vsup 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ self.vsup 2) 0.0]])
          cs1-in (np.array [[gmid-cs1 fug-cs1 (/ self.vsup 2) 0.0]])

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
                  "Wd"    Wdp1  "Wres"  Wres  "Wcap"  Wcap}]

      (.size-step (super) sizing)))

  (defn render [self &optional [mode "ascii"]]
    """
    Prints an ASCII Schematic of the Miller Amplifier courtesy
    https://github.com/Blokkendoos/AACircuit
    """
    (cond [(= mode "ascii")
           (print f"
  #------------------------------------------------------------.        
 VSS                       |                   |               |        
                    MPCM21 +-||             ||-+ MPCM22        |        
                           <-||             ||->               |        
                           +-||--o----------||-+               |        
                           |     |             |               |        
                           o-----'             |            ||-+ MPCS   
                           |                   |            ||->        
                           |                   o------------||-+        
                           |                   |               |        
                           |Y                 X|               |        
                           |                   |   RC_   CC||  |        
                           |                   o--|___|----||--o        
                           |                   |           ||  |        
                           |                   |               |        
                           |                   |               |        
                           |                   |               |        
                           |                   |               |   OUT  
                        ||-+ MND11       MND12 +-||            o----#   
                  INP   ||<-                   ->||   INN      |        
          B        #----||-+                   +-||----#       |        
          #                |        CM         |               |        
          |                '---------o---------'               |        
          |                          |                         |        
          o-----------.              |                         |        
          |           |              |                         |        
          |           |              |                         '        
   MNCM11 +-||        |           ||-+ MNCM12               ||-+ MNCM13 
          ->||        |           ||<-                      ||<-        
          +-||--------o-----------||-+            .---------||-+        
          |           '--------------)------------'            |        
 VDD      |                          |                         |        
  #-------o--------------------------o-------------------------'        
  " )]
          [True (.render (super) mode)])))

(defclass OP1XH035Env [OP1Env]
  """
  Miller Amplifier in XH035 Technology.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [nmos-path None] ^str [pmos-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]

    (setv self.cs   0.85e-15 ; Poly Capacitance per μm^2
          self.rs   100      ; Sheet Resistance in Ω/□
          self.Wres 2e-6     ; Resistor Width in m
          self.Mcap 1e-6     ; Capacitance multiplier
          #_/ )

    (.__init__ (super OP1XH035Env self) :pdk-path pdk-path :ckt-path ckt-path
                                        :nmos-path nmos-path :pmos-path pmos-path
                                        :max-moves max-moves :random-target random-target
                                        :target target :data-log-prefix data-log-prefix
                                        #_/ ))
 
  (defn target-specification ^dict [self &optional ^bool [random False] 
                                                   ^bool [noisy True]]
    """
    Generate a noisy target specification.
    """
    (let [ts {"a_0"          105.0
              "ugbw"         3500000.0
              "pm"           110.0
              "gm"           -45.0
              "sr_r"         2700000.0
              "sr_f"         -2700000.0
              "vn_1Hz"       6.0e-06
              "vn_10Hz"      2.0e-06
              "vn_100Hz"     6.0e-07
              "vn_1kHz"      1.5e-07
              "vn_10kHz"     5.0e-08
              "vn_100kHz"    2.6e-08
              "psrr_n"       120.0
              "psrr_p"       120.0
              "cmrr"         110.0
              "v_il"         0.7
              "v_ih"         3.2
              "v_ol"         0.1
              "v_oh"         3.2
              "i_out_min"    -7e-5
              "i_out_max"    7e-5
              "overshoot_r"  0.0005
              "overshoot_f"  0.0005
              "voff_stat"    0.003
              "voff_sys"     -2.5e-05
              "A"            5.0e-09
              #_/ }
              factor (cond [random (np.abs (np.random.normal 1 0.5))]
                           [noisy  (np.random.normal 1 0.01)]
                           [True   1.0])]
      (dfor (, p v) (.items ts)
        [ p (if noisy (* v factor) v) ]))))

(defclass OP1XH035GeomEnv [OP1XH035Env]
  """
  Miller Amplifier in XH035 Technology.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-target False]
                                 ^dict [target None] ^str [data-log-prefix ""]]

    (.__init__ (super OP1XH035GeomEnv self) :pdk-path pdk-path 
                                            :ckt-path ckt-path
                                            :max-moves max-moves 
                                            :random-target random-target
                                            :target target 
                                            :data-log-prefix data-log-prefix
                                            #_/ )

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
          w-min (list (repeat 0.4e-6 6))  w-max (list (repeat 150e-6 6))
          l-min (list (repeat 0.35e-6 5)) l-max (list (repeat 15e-6 5))
          m-min [1 1 1 1]                 m-max [3 40 10 40]
          self.action-scale-min (np.array (+ w-min l-min m-min))
          self.action-scale-max (np.array (+ w-max l-max m-max))))
 

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
          Mcap                  1 

          sizing {"Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcs"   Lcs1  "Ld"    Ldp1 
                  "Lres"  Lres  "Mcm11" Mcm11 "Mcm12" Mcm12 "Mcm13" Mcm13 
                  "Mcm21" Mcm21 "Mcm22" Mcm22 "Mcs"   Mcs1  "Md"    Mdp1  
                  "Mcap"  Mcap  "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcs"   Wcs1 
                  "Wd"    Wdp1  "Wres"  Wres  "Wcap"  Wcap}]

      (.size-step (super) sizing))))
