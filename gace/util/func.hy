(import os)
(import sys)
(import errno)

(import [functools [partial]])
(import [fractions [Fraction]])
(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [decimal [Decimal]])
(import [operator [itemgetter]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import [.prim [*]])
(import [hace :as ac])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defn scale-value ^float [^float x ^float x-min ^float x-max
                &optional ^float [a -1.0] ^float [b 1.0]]
  """
  Scales a value s.t. x′∈ [a;b], where a = -1.0 and b = 1.0 by default.

              (x - x_min) · (b - a)
    x′ = a + -----------------------
                (x_max - x_min)
  """
  (+ a (/ (* (- x x-min) (- b a)) (- x-max x-min))))

(defn unscale-value ^float [^float x′ ^float x-min ^float x-max
                  &optional ^float [a -1.0] ^float [b 1.0]]
  """
  Scales a value x′∈ [a;b] back to its original, where a = -1.0 and b = 1.0 by
  default.
                (x′- a)
    x = x_min + ------- · (x_max - x_min)
                (b - a)
  """
  (+ x-min (* (/ (- x′ a) (- b a)) (- x-max x-min))))

(defn dec-to-frac ^tuple [^float ratio]
  """
  Turns a float decimal (rounded to nearest .5) into an integer fraction.
  """
  (as-> ratio it (* it 2) (round it) (/ it 2) (str it) 
                 (Decimal it) (Fraction it) 
                 (, it.numerator it.denominator)))

(defn frac-to-dec ^float [^int num ^int den]
  """
  Turns a fraction into a float ratio.
  """
  (/ num den))

(defn ape [t o] 
  """
  Absolute Percentage Error for scalar values.
  """
  (* 100 (/ (np.abs (- t o)) 
            (if (!= t 0) t 1))))

(defn absolute-condition [t c] 
  """
  Returns a function for reward calculation based on the given target `t` and a
  conditional predicate `c`. If the target meets the conditional the reward is
  calculated as: 
            - ape(x)
    r(x) = -e         + 1
  otherwise it is:
    r(x) = - ape(x)
  .
  """
  (let [cn (partial (eval c) t)
      er (partial ape t)]
    (fn [x] 
      (if (cn x) 
         (+ (- (np.exp (- (er x)))) 1) 
         (- (er x))))))

(defn ranged-condition [l u] 
  """
  Returns a function for reward calculation based on the given lower `l` and
  upper `u` bounds. See `absolute-condition` for details.
  """
  (let [er (partial ape (np.abs (- l u)))]
    (fn [x] 
      (if (and (<= l x) (>= u x)) 
         (+ (- (np.exp (- (er x)))) 1)
         (- (er x))))))

(defn ace-constructor [ace-id ace-backend 
                      &optional ^(of list str) [pdk []] 
                                ^str [ckt None]]
  (let [pdk-path ;; Check for PDK
        (cond [(all (lfor p pdk (and p (os.path.exists p)))) pdk]
              [(in "ACE_PDK" os.environ) [(os.environ.get "ACE_PDK")]]
              [(os.path.exists (.format "{}/.ace/{}/pdk" 
                                        (os.path.expanduser "~") 
                                        ace-backend ))
               [(.format "{}/.ace/{}/pdk" (os.path.expanduser "~") 
                                          ace-backend)]]
              [True (raise (FileNotFoundError errno.ENOENT 
                            (os.strerror errno.ENOENT) 
                            (.format "No PDK found for {} and {}"
                                     ace-id ace-backend)))])
    
      ckt-path ;; Check ACE backend Testbench
        (cond [(and ckt (os.path.exists ckt)) ckt]
              [(in "ACE_BACKEND" os.environ) 
               (.format "{}/{}" (os.environ.get "ACE_BACKEND") ace-id)]
              [(os.path.exists (.format "{}/.ace/{}/{}" 
                                        (os.path.expanduser "~") 
                                        ace-backend 
                                        ace-id))
               (.format "{}/.ace/{}/{}" (os.path.expanduser "~") 
                                        ace-backend 
                                        ace-id)]
              [True (raise (FileNotFoundError errno.ENOENT 
                            (os.strerror errno.ENOENT) 
                            (.format "No ACE Testbench found for {} in {}"
                                     ace-id ace-backend)))])

      ace-maker (cond [(.startswith ace-id "op")    ac.single-ended-opamp]
                       [(.startswith ace-id "st")   ac.schmitt-trigger]
                       [(.startswith ace-id "nand") ac.nand-4]
                       [True (raise (NotImplementedError errno.ENOSYS
                            (os.strerror errno.ENOSYS) 
                            (.format "{} is not a valid ACE id."
                                     ace-id)))]) ]
    (fn [] (ace-maker ckt-path :pdk-path pdk-path))))

(defn load-primitive [^str dev-type ^str ace-backend &optional ^str [dev-path ""]]
  (let [device-path (or dev-path (.format "{}/.ace/{}/{}" (os.path.expanduser "~") 
                                          ace-backend dev-type))]
  (if (and device-path (os.path.exists device-path) 
           (all (lfor f ["model.pt" "scale.Y" "scale.X"] 
                        (in f (os.listdir device-path)))))
      (PrimitiveDeviceTs (.format "{}/model.pt" device-path)
                         (.format "{}/scale.X" device-path)
                         (.format "{}/scale.Y" device-path))
      (raise (FileNotFoundError errno.ENOENT 
                            (os.strerror errno.ENOENT) 
                            (.format "No Primitive Device models found at {}."
                                     device-path))))))

(defn starting-point ^(of dict str float) [ace ^bool random ^bool noise]
    """
    Generate a starting point for the agent.
    Arguments:
      ace:        ACE
      [random]:   Random starting point. (default = False)
      [noise]:    Add noise to found starting point. (default = True)
    Returns:      Starting point sizing.
    """
    (let [sizing (if random (ac.random-sizing ace) (ac.initial-sizing ace))]
      (if noise
          (dfor (, p s) (.items sizing) 
                [p (if (or (.startswith p "W") (.startswith p "L")) 
                       (+ s (np.random.normal 0 1e-7)) s)])
          sizing)))

(defn target-distance ^(of tuple np.array) [^(of dict str float) performance 
                                            ^(of dict str float) target
                                            ^(of dict) condition]
  (let [performance-parameters (list (.keys target))
        p-getter (itemgetter #* performance-parameters)

        perf (-> performance (p-getter) (np.array))
        targ (-> target      (p-getter) (np.array))

        dist (/ (np.abs (- perf targ)) targ)

        mask (if condition
                 (lfor (, c p t) 
                    (zip (p-getter condition) perf targ)
                    (c t p))
                 (np.full (len performance-parameters) True))]
      
    (, dist mask perf targ)))

(defn observation ^np.array [^(of dict str float) performance 
                             ^(of dict str float) target]
  """
  Returns observations based on performances and target
  """
  (let [performance-parameters (list (.keys target))
        status-parameters (-> performance (.keys) (set) 
                                          (.difference performance-parameters) 
                                          (list))
        s-getter (if status-parameters (itemgetter #* status-parameters) 
                                       (fn [&rest _] []))
        stat (-> performance (s-getter) (np.array))

        (, dist _ perf targ) (target-distance performance target {})

        obs (-> (, perf targ dist stat) 
                (np.hstack) 
                (np.squeeze) 
                (np.float32))]
      
    (np.nan-to-num obs)))

(defn reward ^float [^(of dict str float) performance
                     ^(of dict str float) target
                     ^(of dict) condition]
  """
  Calculates a reward based on the target and the current perforamnces.
  Arguments:
    performance:  Dictionary with performances.
    target:       Dictionary with target values.
    
  **NOTE**: Both dictionaries must include the keys defined in `params`.
  If no arguments are provided, the current state of the object is used to
  calculate the reward.
  """
  (let [(, loss mask _ _) (target-distance performance target condition)

        cost (+ (* (np.tanh (np.abs loss)) mask) 
                (* (- (** loss 2.0)) (np.invert mask))) ]

       (-> cost (np.nan-to-num) (np.sum))))

(defn info ^(of dict) [^(of dict str float) performance 
                       ^(of dict str float) target 
                       ^(of list str) inputs]
  """
  Returns very useful information about the current state of the circuit,
  simulator and live in general.
  """
  {"output-parameters" (+ (list (sum (zip #* (lfor pp (.keys target)
                                                      (, f"performance_{pp}"
                                                         f"target_{pp}"
                                                         f"distance_{pp}"))) 
                                     (,)))
                          (lfor sp (.keys performance) 
                                   :if (not-in sp (.keys target)) 
                                sp))
   "input-parameters" inputs
   #_/ })

(defn save-state [ace ^str ace-id ^str log-path]
  (ac.dump-state ace :file-name (.format "{}/{}-parameters-{}.json" log-path ace-id
                                         (-> dt (.now) (.strftime "%H%M%S-%y%m%d")))))

(defn save-data [^pd.DataFrame data ^str data-path ^str ace-id]
  (let [time-stamp (-> dt (.now) (.strftime "%H%M%S_%y%m%d"))
        hdf-key (.format "{}_{}" ace-id time-stamp)]
    (-> data (.rename :columns (dfor c data.columns.values
                                     [c (-> c (.replace ":" "-") 
                                              (.replace "." "_"))])) 
             (.to-hdf data-path :key hdf-key :append True :mode "a"))))

(defn technology-data [ace-backend]
  (cond [(= ace-backend "xh035-3V3")
         {"cs"       0.85e-15 ; Poly Capacitance per μm^2
          "rs"       100      ; Sheet Resistance in Ω/□
          "i0"       3e-6     ; Bias Current in A
          "vdd"      3.3      ; Supply Voltage
          "Wres"     2e-6     ; Resistor Width in m
          "Mcap"     1e-6     ; Capacitance multiplier
          "Rc_min"   0.5e3    ; Minimum Compensation Resistor = 500Ω
          "Rc_max"   50e3     ; Minimum Compensation Resistor = 50000Ω
          "Cc_min"   0.5e-12  ; Minimum Compensation Capacitor = 0.5pF
          "Cc_max"   5e-12    ; Minimum Compensation Capacitor = 5pF
          "w_min"    0.4e-6
          "w_max"    150e-6
          "l_min"    0.35e-6
          "l_max"    15e-6
          "gmid_min" 6.0      ; Minimum device efficiency
          "gmid_max" 26.0     ; Maximum device efficiency
          "fug_min"  1e6      ; Minimum device speed
          "fug_max"  1e9      ; Maximum device speed
          #_/ }]
        [(= ace-backend "sky130-1V8")
         {"cs"       0.85e-15 ; Poly Capacitance per μm^2
          "rs"       100      ; Sheet Resistance in Ω/□
          "i0"       3e-6     ; Bias Current in A
          "vdd"      1.8      ; Supply Voltage
          "Wres"     2e-6     ; Resistor Width in m
          "Mcap"     1e-6     ; Capacitance multiplier
          "Rc_min"   0.5e3    ; Minimum Compensation Resistor = 500Ω
          "Rc_max"   50e3     ; Minimum Compensation Resistor = 50000Ω
          "Cc_min"   0.5e-12  ; Minimum Compensation Capacitor = 0.5pF
          "Cc_max"   5e-12    ; Minimum Compensation Capacitor = 5pF
          "w_min"    0.42
          "w_max"    100
          "l_min"    0.15
          "l_max"    100
          "gmid_min" 6.0      ; Minimum device efficiency
          "gmid_max" 26.0     ; Maximum device efficiency
          "fug_min"  1e6      ; Minimum device speed
          "fug_max"  1e9      ; Maximum device speed
          #_/ }]
        [(= ace-backend "gpdk180-1V8")
         {"cs"       0.85e-15 ; Poly Capacitance per μm^2
          "rs"       100      ; Sheet Resistance in Ω/□
          "i0"       3e-6     ; Bias Current in A
          "vdd"      1.8      ; Supply Voltage
          "Wres"     2e-6     ; Resistor Width in m
          "Mcap"     1e-6     ; Capacitance multiplier
          "Rc_min"   0.5e3    ; Minimum Compensation Resistor = 500Ω
          "Rc_max"   50e3     ; Minimum Compensation Resistor = 50000Ω
          "Cc_min"   0.5e-12  ; Minimum Compensation Capacitor = 0.5pF
          "Cc_max"   5e-12    ; Minimum Compensation Capacitor = 5pF
          "w_min"    0.42
          "w_max"    100
          "l_min"    0.18
          "l_max"    20
          "gmid_min" 6.0      ; Minimum device efficiency
          "gmid_max" 26.0     ; Maximum device efficiency
          "fug_min"  1e6      ; Minimum device speed
          "fug_max"  1e9      ; Maximum device speed
          #_/ }]
        [True (raise (NotImplementedError errno.ENOSYS
                            (os.strerror errno.ENOSYS) 
                            (.format "{} is not a valid ACE backend."
                                     ace-backend)))]))
