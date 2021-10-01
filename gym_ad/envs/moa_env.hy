(import os)
(import sys)
(import [functools [partial]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import [skopt [gp-minimize]])

(import gym)
(import [gym.spaces [Dict Box Discrete Tuple]])

(import [.amp_env [AmplifierXH035Env]])
(import [.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass MillerAmpXH035Env [AmplifierXH035Env]
  """
  Derived amplifier class, implementing the Miller Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  Observation Space:
    - See AmplifierXH035Env

  Action Space:
    Continuous Box a: (10,) ∈ [-1.0;1.0]

    Where
    a = [ gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
          fug-cm1  fug-cm2  fug-cm3  fug-dp1 
          mcm1 mcm2 ] 

      mcm1 and mcm2 are the mirror ratios of the corresponding current mirrors.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^str [tech-cfg None]
                                 ^str [nmos-path None] ^str [pmos-path None] 
                                 ^int [max-moves 200]  ^bool [close-target True]
                                 ^float [target-tolerance 1e-3] 
                                 ^dict [target None] ^str [data-log-prefix ""]]
    """
    Constructs a Symmetrical Amplifier Environment with XH035 device models and
    the corresponding netlist.
    Arguments:
      nmos-path:  Prefix path, expects to find `nmos-path/model.pt`, 
                  `nmos-path/scale.X` and `nmos-path/scale.Y` at this location.
      pmos-path:  Same as 'nmos-path', but for PMOS model.

      max-moves:  Maximum amount of steps the agent is allowed to take per
                  episode, before it counts as failed. Default = 200.
      
      close-target: If True (default), on each reset, a random target will be
                    chosen and by bayesian optimization, a location close to it
                    will be found for the starting point of the agent. This
                    increases the reset time significantly.
      target-tolerance: (| target - performance | <= tolerance) ? Success.
      target: Specific target, if given, no random targets will be generated,
              and the agent tries to find the same one over and over again.

    """

    ;; Specify constants as they are defined in the netlist and by the PDK.
    (setv self.vs   0.5       ; in V
          self.cl   5e-12     ; Load Capacitance in F
          self.rl   100e6     ; Load Resistance in Ω
          self.i0   3e-6      ; Bias Current in A
          self.vsup 3.3       ; Supply Voltage in V
          self.fin  1e3       ; Input Frequency in Hz
          self.rs   100       ; Sheet Resistance in Ω/□
          self.cs   0.85e-15) ; Poly Capacitance per μm^2

    ;; Check given paths
    (unless (or pdk-path (not (os.path.exists pdk-path)))
      (raise (FileNotFoundError errno.ENOENT 
                                (os.strerror errno.ENOENT) 
                                pdk-path)))
    (unless (or ckt-path (not (os.path.exists ckt-path)))
      (raise (FileNotFoundError errno.ENOENT 
                                (os.strerror errno.ENOENT) 
                                ckt-path)))
    (unless (or tech-cfg (not (os.path.exists tech-cfg)))
      (raise (FileNotFoundError errno.ENOENT 
                                (os.strerror errno.ENOENT) 
                                tech-cfg)))

    ;; Initialize parent Environment.
    (.__init__ (super MillerAmpXH035Env self) AmplifierID.MILLER
                                              pdk-path ckt-path tech-cfg 
                                              nmos-path pmos-path
                                              max-moves target-tolerance
                                              :close-target close-target
                                              :data-log-prefix data-log-prefix
                                              #_/ )

    ;; Generate random target of None was provided.
    (setv self.same-target  (bool target)
          self.target       (or target (self.target-specification :noisy False)))

    ;; Specify geometric and electric parameters, these have to align with the
    ;; parameters defined in the netlist.
    (setv self.geometric-parameters ["Lcm1" "Lcm2" "Lcs" "Ld" "Lres" 
                                     "Mcm11" "Mcm12" "Mcm13" "Mcm21" "Mcm22" 
                                     "Mcs" "Md" "Mcap" 
                                     "Wcm1" "Wcm2" "Wcs" "Wd" "Wres" "Wcap"]
          self.electric-parameters ["gmid_cm1" "gmid_cm2" "gmid_cs1" "gmid_dp1" 
                                    "fug_cm1" "fug_cm2" "fug_cs1" "fug_dp1" 
                                    "res" "cap" "mcm1" "mcm2"])

    ;; The action space consists of 10 parameters ∈ [0;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 12) 
                                 :dtype np.float32)
          self.action-scale-min (np.array [7.0 7.0 7.0 7.0      ; gm/Id min
                                           1e6 5e5 1e6 1e6      ; fug min
                                           1e3 1e-12            ; C and R min
                                           3.0 10.0])           ; Ratio min
          self.action-scale-max (np.array [17.0 17.0 17.0 17.0  ; gm/Id max
                                           1e9  5e8  1e9  1e9   ; fug max
                                           1e4  10e-12          ; C and R max
                                           7.0  25.0]))         ; Ratio max

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, 176)  :dtype np.float32))
 
    ;; Loss function used for reward calculation. Either write your own, or
    ;; checkout util.Loss for more loss funtions provided with this package. 
    ;; Mean Absolute Percentage Error (MAPE)
    (setv self.loss Loss.MAPE))

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
             res cap mcm2 mcm3 ) (unscale-value action self.action-scale-min 
                                                       self.action-scale-max)
          
          Wres 2e-6
          Lres (* (/ res self.rs) Wres)

          Wcap (* (np.sqrt (/ cap self.cs)) 1e-6)
          Mcap 1
          Mdp1 2
          Mcm21 2
          Mcm22 2
          Mcm11 1
          Mcm12 (round mcm2)
          Mcm13 (round mcm3)
          Mcs1 Mcm13

          vx 1.25 
          i1 (* self.i0 (/ Mcm11 Mcm12))
          i2 (*      i1 (/ Mcm11 Mcm13))

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

      (.size-step (super) (self.clip-sizing sizing))))

  (defn clip-sizing ^dict [self ^dict sizing]
    """
    Clip the chosen values according to PDK Specifiactions.
    """
    (dfor (, p v) (.items sizing)
      [p (cond [(= p "Wcap") (-> v (max 3.5e-6) (min 1e-3))]
               [(= p "Wres") (max v 0.8e-6)]
               [(= p "Lres") (max v 0.8e-6)]
               [(.startswith p "M") (max v 1.0)]
               [(.startswith p "L") (max v 0.35e-6)]
               [(.startswith p "W") (-> v (max 0.4e-6) (min 150e-6))]
               [True v])]))

  (defn target-specification ^dict [self &optional [noisy True]]
    """
    Generate a noisy target specification.
    """
    (let [ts {"A0dB"      100.0
              "ugbw"      (np.array [2500000.0 3000000.0])
              "PM"        80.0
              "GM"        40.0
              "SR-r"      (np.array [2500000.0 3000000.0])
              "SR-f"      (np.array [-2500000.0 -3000000.0])
              "vn-1Hz"    5e-06
              "vn-10Hz"   2e-06
              "vn-100Hz"  5e-07
              "vn-1kHz"   1.5e-07
              "vn-10kHz"  5e-08
              "vn-100kHz" 2.5e-08
              "psrr-n"    80.0
              "psrr-p"    80.0
              "cmrr"      80.0
              "vi-lo"     0.9
              "vi-hi"     2.7
              "vo-lo"     0.1
              "vo-hi"     2.7
              ;"i-out-min" 0.0
              "i-out-max" 7e-5
              "voff-stat" 3e-3
              "voff-syst" 3e-5
              "A"         5.5e-9
              #_/ }]
      (dfor (, p v) (.items ts)
        [ p 
          (if noisy
              (* v (np.random.normal 1 0.01)) v) ])))

  (defn render [self &optional [mode "ascii"]]
    """
    Prints an ASCII Schematic of the Symmetrical Amplifier courtesy
    https://github.com/Blokkendoos/AACircuit
    """
    (cond [(= mode "ascii")
           (print f"Not Implemented")]
          [True (.render (super) mode)])))
