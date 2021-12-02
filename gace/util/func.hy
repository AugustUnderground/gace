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

(defn reward ^float [^(of dict str float) performance
                     ^(of dict str float) target
                     ^(of dict) condition
                     ^(of float) tolerance]
  """
  Calculates a reward based on the target and the current perforamnces.
  Arguments:
    performance:  Dictionary with performances.
    target:       Dictionary with target values.
    target:       Dictionary with binary conditionals 
    
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


(defn action-space ^(of tuple) [ace ^dict dc ^str ace-id  ^int ace-variant]
  """
  Generates an action space for the given ace-id and variant
  """
  (let [ip (input-parameters ace-id ace-variant)

        num-params (fn [p ps] (len (list (filter #%(.startswith %1 p) ps))))

        base-space (Box :low -1.0 :high 1.0 :shape (, (len ip)) :dtype np.float32)

        action-space (cond [(in ace-variant [0 1]) base-space]
                           [(in ace-variant [2 3])
                            (Tuple (, base-space 
                                      (->> ace (ac.simulation-analyses) 
                                           (len) (** 2) (dec) (Discrete))))]
                           [True
                            (raise (NotImplementedError errno.ENOSYS
                                      (os.strerror errno.ENOSYS) 
                                      (.format "No action space for {}-v{}."
                                      ace-id ace-variant)))])

        scale-min (cond [(in ace-variant [0 2])
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
                        [(in ace-variant [1 3])
                         (np.array (lfor p ip (get dc p "min")))]
                        [True
                          (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No action space for {}-v{}."
                                     ace-id ace-variant)))])

        scale-max (cond [(in ace-variant [0 2])
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
                        [(in ace-variant [1 3])
                         (np.array (lfor p ip (get dc p "max")))]
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

(defn input-parameters ^(of list str) [^str ace-id ^int ace-variant]
  """
  Returns a list of input parameter names
  """
  (cond [(and (= ace-id "op1") (in ace-variant [0 2])) 
         [ "gmid-cm1" "gmid-cm2" "gmid-cs1" "gmid-dp1"
           "fug-cm1"  "fug-cm2"  "fug-cs1"  "fug-dp1" 
           "res" "cap" "i1" "i2" ]]
        [(and (= ace-id "op1") (= ace-variant 1)) 
         [ "Ld" "Lcm1"  "Lcm2"  "Lcs"         "Lres"
           "Wd" "Wcm1"  "Wcm2"  "Wcs" "Wcap"  "Wres"
                "Mcm11"         "Mcs"
                "Mcm12" 
                "Mcm13" ]]
        [(and (= ace-id "op2") (in ace-variant [0 2])) 
         [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1"
           "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1" 
           "i1" "i2" ]]
        [(and (= ace-id "op2") (= ace-variant 1))
         [ "Ld" "Lcm1"  "Lcm2" "Lcm3"  
           "Wd" "Wcm1"  "Wcm2" "Wcm3" 
                "Mcm11" "Mcm21"  
                "Mcm12" "Mcm22" ]]
        [(and (= ace-id "op3") (= ace-variant 0))
         [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1"
           "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"
           "i1" "i2" "i3" ]]
        [(and (= ace-id "op3") (= ace-variant 1)) 
         [ "Ld" "Lcm1"  "Lcm2"   "Lcm3" 
           "Wd" "Wcm1"  "Wcm2"   "Wcm3"
                "Mcm11" "Mcm212" "Mcm31" 
                "Mcm12" "Mcm222" "Mcm32" 
                        "Mcm2x1" ]]
        [(and (= ace-id "op4") (= ace-variant 0)) 
         [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1" "gmid-ls1" "gmid-ref" 
           "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"  "fug-ls1"  "fug-ref" 
           "i1" "i2" "i3" ]]
        [(and (= ace-id "op4") (= ace-variant 1)) 
         [ "Ld" "Lcm1"  "Lcm2"  "Lcm3"  "Lc1" "Lr" 
           "Wd" "Wcm1"  "Wcm2"  "Wcm3"  "Wc1" "Wr"
                "Mcm11" "Mcm21"         "Mc1" 
                "Mcm12" "Mcm22" 
                "Mcm13" ]]
        [(and (= ace-id "op5") (= ace-variant 0))
         [ "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-dp1" "gmid-ls1" "gmid-ref"
           "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-dp1"  "fug-ls1"  "fug-ref"
           "i1" "i2" "i3" "i4" ]]
        [(and (= ace-id "op5") (= ace-variant 1)) 
         [ "Ld" "Lcm1"  "Lcm2"   "Lcm3"  "Lc1"  "Lr"
           "Wd" "Wcm1"  "Wcm2"   "Wcm3"  "Wc1"  "Wr"
                "Mcm11" "Mcm212" "Mcm31" "Mc11" 
                "Mcm12" "Mcm222" "Mcm32" "Mc12" 
                "Mcm13" "Mcm2x1" ]]
        [(and (= ace-id "op6") (= ace-variant 0)) 
         [ "gmid-cm1" "gmid-cm2" "gmid-cs1" "gmid-dp1" "gmid-res" "gmid-cap"
           "fug-cm1"  "fug-cm2"  "fug-cs1"  "fug-dp1"  "fug-res"  "fug-cap"
           "i1" "i2" ]]
        [(and (= ace-id "op6") (= ace-variant 1)) 
         [ "Ld" "Lcm1"  "Lcm2"  "Lcs" "Lc1" "Lr1"
           "Wd" "Wcm1"  "Wcm2"  "Wcs" "Wc1" "Wr1"
                "Mcm11"         "Mcs" "Mc1" "Mr1" 
                "Mcm12"
                "Mcm13" ]]
        [(and (= ace-id "op8") (= ace-variant 0)) 
         [ "gmid-dp1" "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-cm4" "gmid-cm5" 
           "fug-dp1"  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-cm4"  "fug-cm5" 
           "i1" "i2" "i3" "i4" ]]
        [(and (= ace-id "op8") (= ace-variant 1)) 
         [ "Ld1" "Lcm1" "Lcm2" "Lcm3" "Lcm4"  "Lcm5"
           "Wd1" "Wcm1" "Wcm2" "Wcm3" "Wcm4"  "Wcm5"
                 "Mcm1" "Mcm2" "Mcm3" "Mcm41" "Mcm51" 
                                      "Mcm42" "Mcm52" 
                                      "Mcm43" "Mcm53" ]]
        [(and (= ace-id "op9") (= ace-variant 0)) 
         [ "gmid-dp1" "gmid-cm1" "gmid-cm2" "gmid-cm3" "gmid-cm4" "gmid-ls1" "gmid-re1" "gmid-re2"
           "fug-dp1"  "fug-cm1"  "fug-cm2"  "fug-cm3"  "fug-cm4"  "fug-ls1"  "fug-re1"  "fug-re2"
           "i1" "i2" "i3" "i4" "i5" "i6" ]]
        [(and (= ace-id "op9") (= ace-variant 1)) 
         [ "Ld1" "Lcm1" "Lcm2" "Lcm3"  "Lcm4"  "Lls1" "Lr1" "Lr2"
           "Wd1" "Wcm1" "Wcm2" "Wcm3"  "Wcm4"  "Wls1" "Wr2" "Wr1"
                 "Mcm1" "Mcm2" "Mcm31" "Mcm41" "Mls1"
                               "Mcm32" "Mcm42"
                               "Mcm33" "Mcm43"
                               "Mcm34" "Mcm44" ]]
        [(and (= ace-id "nand4") (= ace-variant 1)) 
         ["Wn0" "Wp" "Wn2" "Wn1" "Wn3"]]
        [(and (= ace-id "st1") (= ace-variant 1)) 
         ["Wp0" "Wn0" "Wp2" "Wp1" "Wn2" "Wn1"]]
        [True
         (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No parameters for {}-v{}."
                                     ace-id ace-variant)))]))
