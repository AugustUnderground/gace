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
(import [gym.spaces [Dict Box Discrete MultiDiscrete Tuple]])

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

(defn ace-constructor [ace-id ace-backend &optional ^(of list str) [pdk []] ^str [ckt None]]
  (fn [] (ac.make-env ace-id ace-backend :pdk pdk :ckt ckt)))

;;(defn ace-constructor [ace-id ace-backend 
;;                      &optional ^(of list str) [pdk []] 
;;                                ^str [ckt None]]
;;  (let [pdk-path ;; Check for PDK
;;        (cond [(all (lfor p pdk (and p (os.path.exists p)))) pdk]
;;              [(in "ACE_PDK" os.environ) [(os.environ.get "ACE_PDK")]]
;;              [(os.path.exists (.format "{}/.ace/{}/pdk" 
;;                                        (os.path.expanduser "~") 
;;                                        ace-backend ))
;;               [(.format "{}/.ace/{}/pdk" (os.path.expanduser "~") 
;;                                          ace-backend)]]
;;              [True (raise (FileNotFoundError errno.ENOENT 
;;                            (os.strerror errno.ENOENT) 
;;                            (.format "No PDK found for {} and {}"
;;                                     ace-id ace-backend)))])
;;    
;;      ckt-path ;; Check ACE backend Testbench
;;        (cond [(and ckt (os.path.exists ckt)) ckt]
;;              [(in "ACE_BACKEND" os.environ) 
;;               (.format "{}/{}" (os.environ.get "ACE_BACKEND") ace-id)]
;;              [(os.path.exists (.format "{}/.ace/{}/{}" 
;;                                        (os.path.expanduser "~") 
;;                                        ace-backend 
;;                                        ace-id))
;;               (.format "{}/.ace/{}/{}" (os.path.expanduser "~") 
;;                                        ace-backend 
;;                                        ace-id)]
;;              [True (raise (FileNotFoundError errno.ENOENT 
;;                            (os.strerror errno.ENOENT) 
;;                            (.format "No ACE Testbench found for {} in {}"
;;                                     ace-id ace-backend)))])
;;
;;      ace-maker (cond [(.startswith ace-id "op")    ac.single-ended-opamp]
;;                       [(.startswith ace-id "st")   ac.schmitt-trigger]
;;                       [(.startswith ace-id "nand") ac.nand-4]
;;                       [True (raise (NotImplementedError errno.ENOSYS
;;                            (os.strerror errno.ENOSYS) 
;;                            (.format "{} is not a valid ACE id."
;;                                     ace-id)))]) ]
;;    (fn [] (ace-maker ckt-path :pdk-path pdk-path))))

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

(defn observation-shape ^(of tuple int) [ace ^str ace-id ^(of list str) targets]
  """
  Returns a tuple (, int) with the shape of the observation space for the given
  ace backend.
  """
  (, (cond [(.startswith ace-id "op")
            (+ (len (ac.performance-identifiers ace))
               (* 2 (len targets)))]
           [(or (.startswith ace-id "nand") 
                (.startswith ace-id "st"))
            (* 3 (len (ac.performance-identifiers ace)))]
           [True 
            (raise (NotImplementedError errno.ENOSYS
                     (os.strerror errno.ENOSYS) 
                     (.format "No Observation Space shape for given ace id: '{}'."
                              ace-id)))])))

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

(defn absolute-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition]
  """
  Calculates a reward based on the target and the current perforamnces.
  Arguments:
    curr-perf:  Dictionary with performances.
    prev-perf:  Dictionary with performances.
    target:     Dictionary with target values.
    condition:  Dictionary with binary conditionals 
    
  **NOTE**: Both dictionaries must include the keys defined in `params`.
  If no arguments are provided, the current state of the object is used to
  calculate the reward.
  """
  (let [(, loss mask _ _) (target-distance curr-perf target condition)

        cost (+ (* (np.tanh (np.abs loss)) mask) 
                (* (- (** loss 2.0)) (np.invert mask))) ]

    (-> cost (np.nan-to-num) (np.sum))))

(defn relative-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition]
  """
  Calculates a reward based on the relative improvement, compared to previous
  performance. Arguments:
    curr-perf:  Dictionary with performances.
    prev-perf:  Dictionary with performances.
    target:     Dictionary with target values.
    condition:  Dictionary with binary conditionals 
  """
  (let [(, curr-loss curr-mask _ _) (target-distance curr-perf target condition)
        (, prev-loss prev-mask _ _) (target-distance prev-perf target condition)

        curr-cost (+ (* (np.tanh (np.abs curr-loss)) curr-mask) 
                     (* (- (** curr-loss 2.0)) (np.invert curr-mask))) 

        prev-cost (+ (* (np.tanh (np.abs prev-loss)) prev-mask) 
                     (* (- (** prev-loss 2.0)) (np.invert prev-mask))) 

        cost      (- (np.nan-to-num curr-cost) (np.nan-to-num prev-cost))]

    (-> cost (np.sum))))

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


(defn simulation-mask ^(of list str) [ace ^int mask]
  """
  Converts an integer to an ace simulation blocklist.
  """
  (let [analyses      (-> ace (ac.simulation-analyses) (np.array))
        num-analyses  (len analyses)]
    (as-> mask it
          (& it (<< 1 (np.arange num-analyses)))
          (.astype it bool)
          (get analyses it)
          (.tolist it))))

(defn step-v1 ^(of dict str float) [^(of list str) input-parameters
                                    ^np.array action-scale-min 
                                    ^np.array action-scale-max 
                                    ^np.array action]
  """
  Takes a list of input parameters, lower and upper bounds and an action.
  Un-Scales the action according to bounds and returns a sizing dict.
  """
    (dict (zip input-parameters (unscale-value action action-scale-min 
                                                      action-scale-max))))

(defn step-v3 [^(of list str) input-parameters 
               ^(of list str) design-constraints
               ace ^np.array action]
  """
  Takes an relative geometric action.
  """
  (let [cs (ac.current-sizing ace)
        ca (np.array (lfor ip input-parameters (get cs ip)))

        ga (np.array (lfor ip input-parameters 
                           (get design-constraints ip "grid")))

        sa (+ ca (* (- action 1) ga))]
    
    (dict (zip input-parameters sa))))


(defn action-space ^(of tuple) [ace ^dict dc ^str ace-id  ^int ace-variant]
  """
  Generates an action space for the given ace-id and variant

    For variants 0 and 1 the action space is contionous (Box) ∈ [-1; 1], where
    the minimum and maximum are defined by the `design-constraints`' min and max
    values.

    For variants 2 and 3 the action space is discrete (MultiDiscrete) ∈ [0;2]:
    DEC[0], NOP[1], INC[2]. The ∆ for each parameter is defined by the
    `design-constraints`' grid.
  """
  (let [ip (input-parameters ace ace-id ace-variant)

        num-params (fn [p ps] (len (list (filter #%(.startswith %1 p) ps))))

        abs-space (Box :low -1.0 :high 1.0 :shape (, (len ip)) :dtype np.float32)
        rel-space (MultiDiscrete (->> ip (len) (repeat 3) (list)) :dtype np.int32)

        action-space (cond [(in ace-variant [0 1]) abs-space]
                           [(in ace-variant [2 3]) rel-space]
                           [(in ace-variant [4 5])
                            (Tuple (, abs-space 
                                      (->> ace (ac.simulation-analyses) 
                                           (len) (** 2) (dec) (Discrete))))]
                           [(in rel-variant [6 7])
                            (Tuple (, rel-space 
                                      (->> ace (ac.simulation-analyses) 
                                           (len) (** 2) (dec) (Discrete))))]
                           [True
                            (raise (NotImplementedError errno.ENOSYS
                                      (os.strerror errno.ENOSYS) 
                                      (.format "No action space for {}-v{}."
                                      ace-id ace-variant)))])

        scale-min (cond [(in ace-variant [0])   ; Absolute Electrical
                         (np.concatenate 
                          (, (np.repeat (get dc "gmid" "min") 
                                        (num-params "gmid" ip))
                             (np.repeat (get dc "fug" "min")  
                                        (num-params "fug" ip))
                             (np.repeat (get dc "rc" "min")  
                                        (num-params "r" ip))
                             (np.repeat (get dc "cc" "min")  
                                        (num-params "c" ip))
                             (np.repeat (/ (get dc "i0" "init") 3.0) 
                                        (num-params "i" ip))))]
                        [(in ace-variant [1])     ; Absolute Geometrical
                         (np.array (lfor p ip (get dc p "min")))]
                        [(in ace-variant [2 3])   ; relative Electrical and Geometrical
                         None]
                        [True
                          (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No action space for {}-v{}."
                                     ace-id ace-variant)))])

        scale-max (cond [(in ace-variant [0])     ; Absolute Electrical
                        (np.concatenate 
                          (, (np.repeat (get dc "gmid" "max") 
                                        (num-params "gmid" ip))
                             (np.repeat (get dc "fug" "max")  
                                        (num-params "fug" ip))
                             (np.repeat (get dc "rc" "max")  
                                        (num-params "r" ip))
                             (np.repeat (get dc "cc" "max")  
                                        (num-params "c" ip))
                             (np.repeat (/ (get dc "i0" "init") 3.0) 
                                        (num-params "i" ip))))]
                        [(in ace-variant [1])     ; Absolute Geometrical
                         (np.array (lfor p ip (get dc p "max")))]
                        [(in ace-variant [2 3])   ; relative Electrical and Geometrical
                         None]
                        [True
                          (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No action space for {}-v{}."
                                     ace-id ace-variant)))])]

    (, action-space scale-min scale-max)))

(defn design-constraints ^dict [ace]
  """
  Returns a dictionary containing technology constraints.
  """
  (| (ac.parameter-dict ace)
     { "gmid" { "init" 10.0
                "max"  26.0
                "min"  6.0
                "grid" 0.1 }
       "fug" { "init" 1.0e8
               "max"  1.0e9
               "min"  1.0e6
               "grid" 1e4 }
       "rc"  { "init" 5e3
               "max"  50e3
               "min"  0.5e3
               "grid" 0.5e3 }
       "cc" { "init" 1.0e-12
              "max"  5.0e-12
              "min"  0.5e-12
              "grid" 0.2e-12 }
       #_/ }))

(defn input-parameters ^(of list str) [ace ^str ace-id ^int ace-variant]
  """
  Returns a list of input parameter names
  """
  (cond [(in ace-variant [1 3])
         (ac.sizing-identifiers ace)]
        [(and (= ace-id "op1") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cs1" "gmid-dp1"
            "fug-cm1"  "fug-cm2"  "fug-cs1"  "fug-dp1" 
            "res" "cap" "i1" "i2" ]]
        [(and (= ace-id "op2") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1"
            "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1" 
            "i1" "i2" ]]
        [(and (= ace-id "op3") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1"
            "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"
            "i1" "i2" "i3" ]]
        [(and (= ace-id "op4") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1" "gmid-ls1" "gmid-ref" 
            "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"  "fug-ls1"  "fug-ref" 
            "i1" "i2" "i3" ]]
        [(and (= ace-id "op5") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1" "gmid-ls1" "gmid-ref"
            "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"  "fug-ls1"  "fug-ref"
            "i1" "i2" "i3" "i4" ]]
        [(and (= ace-id "op6") (in ace-variant [0 2])) 
          [ "gmid-cm1" "gmid-cm2" "gmid-cs1" "gmid-dp1" "gmid-res" "gmid-cap"
            "fug-cm1"  "fug-cm2"  "fug-cs1"  "fug-dp1"  "fug-res"  "fug-cap"
            "i1" "i2" ]]
        [(and (= ace-id "op8") (in ace-variant [0 2])) 
          [ "gmid-dp1" "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-cm4" "gmid-cm5" 
            "fug-dp1"  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-cm4"  "fug-cm5" 
            "i1" "i2" "i3" "i4" ]]
        [(and (= ace-id "op9") (in ace-variant [0 2])) 
          [ "gmid-dp1" "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-cm4" "gmid-ls1" "gmid-re1" "gmid-re2"
            "fug-dp1"  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-cm4"  "fug-ls1"  "fug-re1"  "fug-re2"
            "i1" "i2" "i3" "i4" "i5" "i6" ]]
        [True 
         (raise (NotImplementedError errno.ENOSYS
                                     (os.strerror errno.ENOSYS) 
                                     (.format "No parameters for {}-v{}."
                                     ace-id ace-variant)))]))
