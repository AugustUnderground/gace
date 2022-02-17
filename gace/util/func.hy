(import os)
(import sys)
(import errno)

(import [functools [partial]])
(import [fractions [Fraction]])
(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [decimal [Decimal]])

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

(defn ace-constructor [^str ace-id ^str ace-backend
        &optional ^(of list str) [pdk []] ^str [ckt None] ^int [num-envs 1]]
  """
  Meta function for (re-)creating environments.
  """
  (if (> num-envs 1)
      `(ac.make-same-env-pool ~num-envs ~ace-id ~ace-backend :pdk ~pdk :ckt ~ckt)
      `(ac.make-env ~ace-id ~ace-backend :pdk ~pdk :ckt ~ckt)))

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
  (let [performance-parameters    (-> target (.keys) (list) (sorted))

        perf (np.array (lfor pp performance-parameters (get performance pp)))
        targ (np.array (lfor pp performance-parameters (get target pp)))
        crit (if condition (lfor pp performance-parameters (get condition pp)) [])
        
        ;dist (/ (np.abs (- perf targ)) targ)
        ;dist (/ (- perf targ) targ)
        dist (/ (- (np.abs perf) (np.abs targ)) (np.abs targ))

        mask (np.array (lfor (, c p t) (zip crit perf targ) (c t p)))
        #_/ ]
      
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
                             ^(of dict str float) target
                             ^int steps
                             ^int max-steps]
  """
  Returns observations: Performance, Target, Distance, Operating Point
  """
  (let [operatingpoint-parameters (->> performance (.keys) (filter #%(in ":" %1)) 
                                                   (list) (sorted))
        oper (np.array (lfor op operatingpoint-parameters (get performance op)))
        (, dist _ perf targ) (target-distance performance target {})
        step (np.array [steps max-steps])

        obs (-> (, perf targ dist oper step) 
                (np.hstack) 
                (np.squeeze) 
                (np.float32))]
      
    (np.nan-to-num obs)))

(defn absolute-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition
                              ^int steps ^int max-steps
                              &optional ^float [bonus 10.0]]
  """
  Calculates a reward based on the target and the current perforamnces.
  Arguments:
    curr-perf:  Dictionary with performances.
    prev-perf:  Dictionary with performances.
    target:     Dictionary with target values.
    condition:  Dictionary with binary conditionals 
    steps:      Number of steps taken in the environment.
  """
  (let [(, loss mask _ _) (target-distance curr-perf target condition)

        cost (+ (* (np.tanh (np.abs loss)) mask) 
                (* (- (np.abs loss)) (np.invert mask))) 

        finish-bonus (* (and (np.all mask) (<= steps max-steps)) bonus)
      #_/ ]

    (-> cost (np.nan-to-num) (np.sum) (+ finish-bonus))))

(defn simple-reward ^float [^(of dict str float) curr-perf
                            ^(of dict str float) prev-perf
                            ^(of dict str float) target
                            ^(of dict) condition
                            ^int steps ^int max-steps
                            &optional ^float [improv-fact 2.0]
                                      ^float [bonus 10.0]]
  """
  Calculates a reward based on the relative improvement, compared to previous
  performance. Arguments:
    curr-perf:  Dictionary with performances.
    prev-perf:  Dictionary with performances.
    target:     Dictionary with target values.
    condition:  Dictionary with binary conditionals .
    steps:      Number of steps taken in the environment.
  """
  (let [(, curr-dist curr-mask _ _) (target-distance curr-perf target condition)
        (, prev-dist prev-mask _ _) (target-distance prev-perf target condition)
  
        better (| (& (np.invert prev-mask) curr-mask)
                  (& (np.invert curr-mask)
                     (np.invert prev-mask)
                     (< curr-dist prev-dist)))

        stayed (& prev-mask curr-mask)

        worse (& (np.invert prev-mask) 
                 (np.invert curr-mask)
                 (>= curr-dist prev-dist))

        worst (& prev-mask (np.invert curr-mask))

        ;simple (+ (-> better (.astype float) (* 1.0)) 
        ;          (-> stayed (.astype float) (* steps) (np.clip 1.0 5.0))
        ;          (-> worse (.astype float) (* steps) (np.clip 1.0 10.0) (-))
        ;          (-> worst (.astype float) (* steps) (-))) 

        simple (+ (-> better (.astype float) (* 1.0)) 
                  (-> stayed (.astype float) (* 2.0))
                  (-> worse (.astype float) (* -1.0))
                  (-> worst (.astype float) (* -3.0)))
        
        finish-bonus (* (and (np.all curr-mask) (<= steps max-steps)) bonus)
        #_/ ]

    (-> simple (.astype float) (np.sum) (np.nan-to-num) (+ finish-bonus))))
    ;(-> simple (.astype float) (np.sum) (- steps) (np.nan-to-num))))

(defn relative-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition
                              ^int steps ^int max-steps
                              &optional ^float [improv-fact 10.0]
                                        ^float [bonus 10.0]]
  """
  Calculates a reward based on the relative improvement, compared to previous
  performance. Arguments:
    curr-perf:  Dictionary with performances.
    prev-perf:  Dictionary with performances.
    target:     Dictionary with target values.
    condition:  Dictionary with binary conditionals .
    steps:      Number of steps taken in the environment.
  """
  (let [(, curr-dist curr-mask _ _) (target-distance curr-perf target condition)
        (, prev-dist prev-mask _ _) (target-distance prev-perf target condition)

        curr-rew (+ (* (np.tanh (np.abs curr-dist)) curr-mask) 
                     (* (- (np.abs curr-dist)) (np.invert curr-mask))) 
                  
        prev-rew (+ (* (np.tanh (np.abs prev-dist)) prev-mask) 
                     (* (- (np.abs prev-dist)) (np.invert prev-mask))) 

        improv-mask (>= curr-rew prev-rew)

                   ;; Improvement but not reached target
        sum-rew (+ (* (- curr-rew prev-rew)
                      (& improv-mask (np.invert curr-mask)))

                   ;; Improvement above target
                   (* (- curr-rew prev-rew)
                      (& improv-mask curr-mask)
                      improv-fact)

                   ;; Decline before reaching target
                   (* (- curr-rew prev-rew) 
                      (& (np.invert improv-mask) 
                         (np.invert curr-mask) 
                         (np.invert prev-mask))
                      improv-fact)

                   ;; Decline after reaching target and still above target
                   (* (np.abs (- curr-rew prev-rew))
                      (& (np.invert improv-mask) curr-mask prev-mask)
                      (/ improv-fact 2))

                   ;; Decline after reaching target and below target now
                   (* (- curr-rew prev-rew)
                      (& (np.invert improv-mask) 
                         (np.invert curr-mask) 
                         prev-mask)
                      improv-fact)) 

        finish-bonus (* (and (np.all curr-mask) (<= steps max-steps)) bonus)
        #_/ ]

    ;(-> sum-rew (np.sum) (- steps) (np.nan-to-num))))
    (-> sum-rew (np.sum) (np.nan-to-num) (+ finish-bonus))))

(defn info ^(of dict) [^(of dict str float) performance 
                       ^(of dict str float) target 
                       ^(of list str) inputs]
  """
  Returns very useful information about the current state of the circuit,
  simulator and live in general.
  """
  {"output-parameters" (+ (list (sum (zip #* (lfor pp (-> target (.keys) (list) (sorted))
                                                      (, f"performance_{pp}"
                                                         f"target_{pp}"
                                                         f"distance_{pp}"))) 
                                     (,)))
                          (->> performance (.keys) (filter #%(in ":" %1)) 
                                           (list) (sorted))
                          ["steps" "max_steps"])
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

(defn sizing-step ^(of dict str float) [^(of list str) input-parameters
                                    ^np.array action-scale-min 
                                    ^np.array action-scale-max 
                                    ^np.array action]
  """
  Takes a list of input parameters, lower and upper bounds and an action.
  Un-Scales the action according to bounds and returns a sizing dict.
  """
    (dict (zip input-parameters (unscale-value action action-scale-min 
                                                      action-scale-max))))

(defn sizing-step-relative [^(of list str) input-parameters 
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
  (let [;_ace (if (ac.is-pool-env ace) (first ace.envs) ace)
        ip (input-parameters ace ace-id ace-variant)

        num-params (fn [p ps] (len (list (filter #%(.startswith %1 p) ps))))

        abs-space (Box :low -1.0 :high 1.0 :shape (, (len ip)) :dtype np.float32)
        rel-space (MultiDiscrete (->> ip (len) (repeat 3) (list)) :dtype np.int32)

        space (cond [(in ace-variant [0 1]) abs-space]
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
                             (np.repeat (* (get dc "i0" "init") 5.0) 
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

    (, space scale-min scale-max)))

(defn design-constraints ^dict [ace]
  """
  Returns a dictionary containing technology constraints.
  """
  (-> ace ;(ac.is-pool-env) (if (first ace.envs) ace)
    (ac.parameter-dict)
    (| { "gmid" { "init" 10.0
                  "max"  25.0
                  "min"  5.0
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
         #_/ })))

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
