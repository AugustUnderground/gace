(import os)
(import sys)
(import errno)

(import [functools [partial]])
(import [fractions [Fraction]])
(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [decimal [Decimal]])

(import [numpy :as np])
(import [csv [DictWriter]])
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
        &optional ^(of list str) [pdk []] ^str [ckt None] ^str [sim None]
                  ^int [num-envs 1]]
  """
  Meta function for (re-)creating environments.
  """
  (if (> num-envs 1)
      `(ac.make-same-env-pool ~num-envs ~ace-id ~ace-backend :pdk ~pdk :ckt ~ckt :sims ~sim)
      `(ac.make-env ~ace-id ~ace-backend :pdk ~pdk :ckt ~ckt :sim ~sim)))

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

(defn starting-point ^(of dict str float) [ace ^int ace-variant ^int reset-count
      ^int num-steps ^int max-steps ^dict constraints ^bool random ^bool noise]
    """
    Generate a starting point for the agent.
    Arguments:
      ace:     ACE
      variant:     ace varaint [0 .. 4]
      reset-count: Determines the random-ness
      random:      Random starting point. (default = False)
      noise:       Add noise to found starting point. (default = True)
    Returns: Starting point sizing.
    """
    (cond [(in ace-variant [0 1]) 
           (ac.random-sizing ace)]
          [(or (<= num-steps 0) (>= num-steps max-steps))
           (ac.initial-sizing ace)]
          [True
           (let [sizing (if (or (> reset-count 50) random) 
                           (ac.random-sizing ace) 
                           (ac.initial-sizing ace))]
              (if noise
                  (dfor (, p s) (.items sizing) 
                        [p (cond [(or (.startswith p "W") (.startswith p "L"))
                                  (let [l (* (get constraints p "grid") 10.0)
                                        m (+ (* l (np.tanh (- (/ reset-count 35.0) 2.0))) l)
                                        n (np.random.normal 0.0 (np.abs m))]
                                    (np.abs (+ s n)))]
                                 [(.startswith p "M")
                                  (let [vals (np.arange (get constraints p "min")
                                                        (get constraints p "max")
                                                        (get constraints p "grid"))
                                        choices (if (not vals.size)
                                                    (np.array [s]) vals)
                                        weights (+ (np.full (len vals) 
                                                            (+ (- (np.exp (/ (- reset-count) 
                                                                             25.0))) 
                                                               1.0))
                                                   (* (np.random.rand (len vals)) 1e-3))
                                        w       (np.where (= vals s) 1.0 weights)
                                        probs   (/ w (.sum w)) ]
                                    (if (not vals.size) s
                                      (-> vals (np.random.choice 1 :p probs) (.item))))]
                                 [True s])])
                sizing)) ]))

(defn target-distance ^(of tuple np.array) [^(of dict str float) performance 
                                            ^(of dict str float) target
                                            ^(of dict) condition]
  """
  Calculates normalized distance between performance and target and retruns a
  mask, which performances were met.
  """
  (let [targets    (-> target (.keys) (list) (sorted))

        ;; Convert Gain from dB into absolute
        ;perf (np.array (lfor pp targets 
        ;                     (if (= pp "a_0")
        ;                         (np.power 10 (/ (get performance pp) 20.0))
        ;                         (get performance pp))))

        ;targ (np.array (lfor pp targets 
        ;                     (if (= pp "a_0")
        ;                         (np.power 10 (/ (get target pp) 20.0))
        ;                         (get target pp))))

        ;perf (np.nan-to-num (np.array (lfor pp targets (get performance pp))))
        ;targ (np.nan-to-num (np.array (lfor pp targets (get target pp))))

        perf (np.array (lfor pp targets (get performance pp)))
        targ (np.array (lfor pp targets (get target pp)))
        crit (if condition (lfor pp targets (get condition pp)) [])
        
        ;dist (/ (np.abs (- perf targ)) targ)
        ;dist (/ (- perf targ) targ)
        dist (/ (- (np.abs perf) (np.abs targ)) (np.abs targ))

        mask (np.array (lfor (, c t p) (zip crit targ perf) (c t p)))
        #_/ ]
      
    (, dist mask)))

(defn sorted-parameters [^(of list str) performance]
  """
  Returns a sorted list of parameters.
  """
  (let [;; Operating Point Parameters
        op (->> performance (filter #%(in ":" %1)) (list) (sorted))
        ;; Offset Contributions
        os (->> performance (filter #%(in "/" %1)) (list) (sorted))
        ;; Node Voltages
        nd (->> performance
                (filter #%(and (.isupper (first %1)) (!= %1 "A") (<= (len %1) 2))) 
                (list) (sorted))
        ;; Performance Parameters
        pf (->> performance (filter #%(not-in %1 (+ op os nd))) (list) (sorted))
        sp ["operating-point" "offset-contribution" "node-voltages" "performance"]]
    (dict (zip sp [op os nd pf]))))

(defn observation-shape ^(of tuple int) [ace ^str ace-id ^(of list str) targets]
  """
  Returns a tuple (, int) with the shape of the observation space for the given
  ace backend.
  """
  (, (cond [(.startswith ace-id "op")
            (+ (len (ac.performance-identifiers ace))
               (* 2 (len targets)) 
               2)]
           [(or (.startswith ace-id "nand") 
                (.startswith ace-id "st"))
            (+ (* 3 (len (ac.performance-identifiers ace))) 
               2)]
           [True 
            (raise (NotImplementedError errno.ENOSYS
                     (os.strerror errno.ENOSYS) 
                     (.format "No Observation Space shape for given ace id: '{}'."
                              ace-id)))])))

(defn observation ^np.array [^(of dict str float) performance 
                             ^(of dict str float) target
                             ^int steps ^int max-steps]
  """
  Returns observations: Performance, Target, Distance, Operating Point
  """
  (let [sp (sorted-parameters (.keys performance))
        (, op os nd pf) (lfor p ["operating-point" "offset-contribution" 
                                 "node-voltages" "performance"] 
                              (get sp p))

        ;; Target Parameters
        tg (-> target (.keys) (list) (sorted))

        ;; Get Parameters enuring same order every time
        oper (np.array (lfor p op (get performance p)))
        offs (np.array (lfor p os (get performance p)))
        nods (np.array (lfor p nd (get performance p)))
        perf (np.array (lfor p pf (get performance p)))
        targ (np.array (lfor p tg (get target p)))

        ;; Obtain distance to target
        (, dist _ ) (target-distance performance target {})

        ;; Step count and maximum number of steps
        step (np.array [steps max-steps])

        ;; Join
        obs (-> (, perf targ dist oper offs nods step) 
                (np.hstack) 
                (np.squeeze) 
                (np.float32))]
      
    ;; Convert NaNs and +/- Infs
    (np.nan-to-num obs)))

(defn absolute-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition
                              ^(of dict str float) curr-sizing
                              ^(of dict str float) set-sizing
                              ^(of dict str float) last-action
                              ^int steps ^int max-steps
                              &optional ^float [bonus 1.5]]
  """
  Calculates a reward based on the target and the current perforamnces.
  Arguments:
    curr-perf:    Dictionary with performances.
    prev-perf:    Dictionary with performances.
    target:       Dictionary with target values.
    condition:    Dictionary with binary conditionals .
    curr-sizing:  The actual sizing of the netlist.
    set-sizing:   The sizing set directly or via conversion.
    last-action:  The last action performed.
    steps:        Number of steps taken in the environment.
    max-steps:    Maximum number of steps allowed.
  """
  (let [(, dist mask) (target-distance curr-perf target condition)

        d             (+ (* (np.abs dist) mask) 
                         (* (- (np.abs dist)) (np.invert mask)))

        l             (+ (- (np.exp (- (np.sum d)))) 1.0)
        perf-loss     (np.where (< l 0.0) (- (np.log (np.abs l))) l)

        ;perf-loss     (+ (- (np.exp (- d))) 1.0)

        ;perf-loss     (+ (* (np.tanh (np.abs dist)) mask) 
        ;                 (* (- (np.abs dist)) (np.invert mask))) 

        ;last-act      (dfor (, k v) (.items last-action) 
        ;                    [k (cond [(.endswith k ":fug") (np.power 10 v)]
        ;                             [(.endswith k ":id") (* v 1e-6)] 
        ;                             [True v])])

        action-loss   (-> (lfor a (.keys last-action)
                                  (/ (-  (get last-action a) (get curr-perf a)) 
                                     (get curr-perf a)))
                          (np.array) (np.sum) (* 1.0e-3))

        sizing-loss   (-> (lfor s (.keys curr-sizing)
                                  (/ (- (get set-sizing s) (get curr-sizing s)) 
                                     (get curr-sizing s)))
                          (np.array) (np.sum) (* 1.0e-4))

        step-loss     (* steps 5.0e-2)
        step-bonus    (* (- max-steps steps) bonus 1.0e-1)

        finish-bonus  (* (np.all mask) 3.0 bonus)

        ;finish-fail   (* (or (np.all (np.invert mask)) (>= steps max-steps)) bonus)
        finish-fail   (* (or (np.any (np.invert mask)) 
                            (>= steps max-steps)) 
                            3.0 bonus)

        loss          (* (np.tanh (/ (- perf-loss action-loss step-loss) 10.0)) 
                         10.0)

        reward        (-> loss (- finish-fail) (+ finish-bonus) 
                               (np.minimum 25.0) (np.maximum (- 25.0))
                               (.item))

                                ;(-> perf-loss 
                                ;    (- action-loss)
                                ;    (- sizing-loss)
                                ;    (+ finish-bonus)
                                ;    (- finish-fail)
                                ;    (np.maximum (- 25.0))
                                ;    (.item))
        #_/ ]
    (when (np.isnan reward) 
      (setv tk   (-> target (.keys) (list) (sorted))
            perf (dfor pp tk [ pp (get curr-perf pp) ]))
      (print "")
      (pp perf)
      (print last-action)
      (print ""))
    reward))

(defn simple-reward ^float [^(of dict str float) curr-perf
                            ^(of dict str float) prev-perf
                            ^(of dict str float) target
                            ^(of dict) condition
                            ^(of dict str float) curr-sizing
                            ^(of dict str float) set-sizing
                            ^(of dict str float) last-action
                            ^int steps ^int max-steps
                            &optional float [tolerance 1e-3]
                            #_/ ]
  """
  Calculates a reward based on the relative improvement, compared to previous
  performance. Arguments:
    curr-perf:    Dictionary with performances.
    prev-perf:    Dictionary with performances.
    target:       Dictionary with target values.
    condition:    Dictionary with binary conditionals .
    curr-sizing:  The actual sizing of the netlist.
    set-sizing:   The sizing set directly or via conversion.
    last-action:  The last action performed.
    steps:        Number of steps taken in the environment.
    max-steps:    Maximum number of steps allowed.
  """
  (let [(, _ curr-mask) (target-distance curr-perf target condition)]
    (if (and (< steps max-steps) (np.all curr-mask)) 0.0 (- 1.0))))

(defn relative-reward ^float [^(of dict str float) curr-perf
                              ^(of dict str float) prev-perf
                              ^(of dict str float) target
                              ^(of dict) condition
                              ^(of dict str float) curr-sizing
                              ^(of dict str float) set-sizing
                              ^(of dict str float) last-action
                              ^int steps ^int max-steps
                              &optional ^float [bonus 10.0]
                                        ^float [improv-fact 2.0]]
  """
  Calculates a reward based on the relative improvement, compared to previous
  performance. Arguments:
    curr-perf:    Dictionary with performances.
    prev-perf:    Dictionary with performances.
    target:       Dictionary with target values.
    condition:    Dictionary with binary conditionals .
    curr-sizing:  The actual sizing of the netlist.
    set-sizing:   The sizing set directly or via conversion.
    last-action:  The last action performed.
    steps:        Number of steps taken in the environment.
    max-steps:    Maximum number of steps allowed.
  """
  (let [(, curr-dist curr-mask) (target-distance curr-perf target condition)
        (, prev-dist prev-mask) (target-distance prev-perf target condition)

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

        last-act    (dfor (, k v) (.items last-action) 
                          [k (cond [(.endswith k ":fug") (np.power 10 v)]
                                   [(.endswith k ":id") (* v 1e-6)] 
                                   [True v])])

        act-loss (-> (lfor a (.keys last-act)
                             (/ (-  (get last-act a) (get curr-perf a)) 
                                (get curr-perf a)))
                     (np.array) (np.sum))

        finish-bonus (* (and (np.all curr-mask) (<= steps max-steps)) bonus)
        #_/ ]

    ;(-> sum-rew (np.sum) (- steps) (np.nan-to-num))))
    (-> sum-rew (np.sum) (np.nan-to-num) (+ finish-bonus) (- act-loss))))

(defn info ^(of dict) [^(of dict str float) performance 
                       ^(of dict str float) target 
                       ^(of list str) inputs]
  """
  Returns very useful information about the current state of the circuit,
  simulator and live in general.
  """
  (let [sp (sorted-parameters (.keys performance))
        (, op os nd pf) (lfor p ["operating-point" "offset-contribution" 
                                 "node-voltages" "performance"] 
                              (get sp p))
        ;; Target Parameters
        tg (->> target (.keys) (map #%(.format "target_{}" %1)) (list) (sorted))
        dt (lfor t (.keys target) (.format "delta_{}" t))]
  {"observations" (+ pf tg dt op os nd ["steps" "max-steps"])
   "actions" inputs}))

(defn initialize-data-logger ^(of dict str str) [ace-env ^int ace-variant 
        ^(of dict str float) target ^(of list str) inputs ^str log-path]
  (let [p  (ac.performance-identifiers ace-env) 
        ;ph (+ ["episode" "step"] (sorted (list (reduce + (.values (sorted-parameters p))))))
        ph (+ ["episode" "step"] (sorted (list (.keys target)))
              (if (in ace-variant [0 2]) (sorted inputs) []))
        eh ["episode" "step" "reward"]
        sh (+ ["episode" "step"] (sorted (ac.sizing-identifiers ace-env)))
        th (+ ["episode"] (sorted (list (.keys target))))
        pp (.format "{}/performance.csv" log-path)
        ep (.format "{}/environment.csv" log-path)
        sp (.format "{}/sizing.csv"      log-path)
        tp (.format "{}/target.csv"      log-path) 
        pk "performance"
        ek "environment"
        sk "sizing"
        tk "target"
        hs [ph eh sh th]
        ps [pp ep sp tp]
        ks [pk ek sk tk]
        #_/ ]

    (os.makedirs log-path :exist-ok True)

    (dfor (, k p h) (zip ks ps hs )
      :do (with [f (open p "w" :newline "\n")]
            (setv dw (DictWriter f :fieldnames h))
            (.writeheader dw))
        [k p])))

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

(defn sizing-step ^(of dict str float) [^(of list str) inputs
                                    ^np.array action-scale-min 
                                    ^np.array action-scale-max 
                                    ^np.array action]
  """
  Takes a list of input parameters, lower and upper bounds and an action.
  Un-Scales the action according to bounds and returns a sizing dict.
  """
    (dict (zip inputs (unscale-value action action-scale-min action-scale-max))))

(defn sizing-step-relative [^(of list str) inputs 
               ^(of list str) design-constraints
               ace ^np.array action]
  """
  Takes an relative geometric action.
  """
  (let [cs (ac.current-sizing ace)
        ca (np.array (lfor ip inputs (get cs ip)))

        ga (np.array (lfor ip inputs 
                           (get design-constraints ip "grid")))

        sa (+ ca (* (- action 1) ga))]
    
    (dict (zip inputs sa))))

(defn discrete-step ^(of tuple np.array float bool dict) [ ^(of list str) inputs
          ^(of list str) design-constraints ^np.array action-scale-min
          ^np.array action-scale-max ace sizing-fun 
          ^int num-gmid ^int num-fug ^int num-ib
          ^int action-idx 
          &optional [blocklist []] ]
    """
    Takes an array of descrete electric parameters for each building block and 
    converts them to sizing parameters for each parameter specified in the
    netlist. 
    """
    (if (= 0 action-idx)
        (ac.current-sizing ace)
        (let [current-performance (ac.current-performance ace)

              current-params (np.array (lfor p inputs
                                             (cond [(.endswith p ":fug") 
                                                    (np.log10 (get current-performance p))]
                                                   [(.endswith p ":id") 
                                                    (* (get current-performance p) 1.0e6)]
                                                   [True (get current-performance p)])))

              grid-action (np.array 
                            (+ (-> design-constraints (get "gmoverid" "grid") 
                                                     (repeat num-gmid) 
                                                     (list))
                               (-> design-constraints (get "fug" "grid") 
                                                      (repeat num-fug) 
                                                      (list))
                               (-> 1.0 (repeat num-ib) (list))))

              (, up dn) (np.array-split (get (np.eye (* 2 (len inputs))) 
                                             (- action-idx 1)) 2)
              
              action (-> (- up dn)
                         (* grid-action)
                         (+ current-params) 
                         (np.maximum action-scale-min)
                         (np.minimum action-scale-max)
                         (scale-value action-scale-min 
                                      action-scale-max))

              #_/ ]

          (sizing-fun action :blocklist blocklist))))

(defn action-space ^(of tuple) [ace ^dict dc ^str ace-id  ^int ace-variant]
  """
  Generates an action space for the given ace-id and variant

    For variants 0 and 1 the action space is contionous (Box) ∈ [-1; 1], where
    the minimum and maximum are defined by the `design-constraints`' min and max
    values.

    For variants 2 and 3 the action is `Discrete` with num-inputs * 2 + 1
    actions. The first one is always `Do nothing` the rest ist +1 -1.
  """
  (let [ip (input-parameters ace ace-id ace-variant)

        num-params (fn [p ps] (len (list (filter #%(.endswith %1 (+ ":" p)) ps))))
        num-caps (fn [ps] (len (list (filter #%(in "cap" %1) ps))))
        num-ress (fn [ps] (len (list (filter #%(in "res" %1) ps))))

        abs-space (Box :low -1.0 :high 1.0 :shape (, (len ip)) :dtype np.float32)
        ;rel-space (MultiDiscrete (->> ip (len) (repeat 3) (list)) :dtype np.int32)
        rel-space (Discrete (-> ip (len) (* 2) (+ 1) (int)))

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

        scale-min (cond [(in ace-variant [0 2])   ; Electrical
                         (np.concatenate 
                          (, (np.repeat (get dc "gmoverid" "min") 
                                        (num-params "gmoverid" ip))
                             (np.repeat (get dc "fug" "min")  
                                        (num-params "fug" ip))
                             (np.repeat (get dc "rc" "min") (num-ress ip))
                             (np.repeat (get dc "cc" "min") (num-caps ip))
                             ;(np.repeat (get dc "ib" "min") (num-params "id" ip))
                             (get dc "ib" "min")
                             ))]
                        [(in ace-variant [1])     ; Absolute Geometrical
                         (np.array (lfor p ip (get dc p "min")))]
                        [(in ace-variant [3])     ; relative Geometrical
                         None]
                        [True
                          (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No action space for {}-v{}."
                                     ace-id ace-variant)))])

        scale-max (cond [(in ace-variant [0 2])     ; Electrical
                        (np.concatenate 
                          (, (np.repeat (get dc "gmoverid" "max") 
                                        (num-params "gmoverid" ip))
                             (np.repeat (get dc "fug" "max")  
                                        (num-params "fug" ip))
                             (np.repeat (get dc "rc" "max") (num-ress ip))
                             (np.repeat (get dc "cc" "max") (num-caps ip))
                             ;(np.repeat (get dc "ib" "max") (num-params "id" ip))
                             (get dc "ib" "max")
                             ))]
                        [(in ace-variant [1])     ; Absolute Geometrical
                         (np.array (lfor p ip (get dc p "max")))]
                        [(in ace-variant [3])     ; relative Geometrical
                         None]
                        [True
                          (raise (NotImplementedError errno.ENOSYS
                                    (os.strerror errno.ENOSYS) 
                                    (.format "No action space for {}-v{}."
                                     ace-id ace-variant)))])]

    (, space scale-min scale-max)))

(defn cap2wid [^str ace-backend ^float C]
  """
  Convert a capacitance value to width
  """
  (cond [(= ace-backend "xh035-3V3")
          (- (/ (np.sqrt (+ 7.056e-21 (* 3.4e-3 C))) 1.7e-3) 49e-9)]
        [True 
         (* (np.sqrt (/ C 5e-12)) 1) ]))

(defn res2len [^str ace-backend ^float R]
  """
  Convert a resistance value to length
  """
  (cond [(= ace-backend "xh035-3V3")
         (+ (* 9.7e-9 R) 3.819e-7)]
        [True 
         (* (/ R 100) 2.0e-6)]))

(defn scale-sizing [^(of dict str float) sizing ^float factor]
  """
  Scales sizing for netlist.
  """
  (dfor (, k v) (.items sizing) 
        [k (if (or (.startswith k "W") (.startswith k "L")) 
               (* v (/ 1 factor)) v)]))

(defn design-constraints ^dict [ace ^str ace-id ^str ace-backend]
  """
  Returns a dictionary containing technology constraints.
  """
  (-> ace (ac.parameter-dict)
    (| { "gmoverid" { "init" 10.0
                      "max"  15.0
                      "min"  5.0
                      "grid" 0.5 }
         "fug" { "init" 7.5 ;; 1.0e7.5
                        "max"  9.0 ;; 1.0e9.0
                        "min"  6.0 ;; 1.0e6.0
                        "grid" 0.1 }
                  ;(cond [(= ace-backend "xh035-3V3")
                  ;    { "init" 7.5 ;; 1.0e7.5
                  ;      "max"  9.0 ;; 1.0e9.0
                  ;      "min"  6.0 ;; 1.0e6.0
                  ;      "grid" 0.1 }] 
                  ;   [(= ace-backend "xh018-1V8")
                  ;    { "init" 7.5 
                  ;      "max"  9.3 
                  ;      "min"  5.0
                  ;      "grid" 0.1 }]
                  ;   [True
                  ;    { "init" 7.5 ;; 1.0e7.5
                  ;      "max"  9.0 ;; 1.0e9.0
                  ;      "min"  6.0 ;; 1.0e6.0
                  ;      "grid" 0.1 }])
         "ib"  (cond [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op1"))
                      { "init" [15.0 60.0]
                        "min" [10.0 40.0]
                        "max" [30.0 80.0]
                        "grid" 1.0 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op2"))
                      { "init" [6.0 12.0]
                        "min" [1.0 1.0]
                        "max" [30.0 30.0]
                        "grid" 1.0 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op3"))
                      { "init" [6.0 12.0 12.0]
                        "min" [1.0 1.0 1.0]
                        "max" [30.0 30.0 30.0]
                        "grid" 1.0 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op4"))
                      { "init" [6.0 12.0 12.0]
                        "min" [1.0 1.0 1.0]
                        "max" [30.0 60.0 60.0]
                        "grid" 1.0 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op5"))
                      { "init" [6.0 12.0 12.0 30.0]
                        "min" [1.0 1.0 1.0 3.0]
                        "max" [30.0 60.0 60.0 90.0]
                        "grid" 1.0 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op6"))
                      { "init" [6.0 12.0]
                        "min" [1.0 3.0]
                        "max" [30.0 90.0]
                        "grid" 1.0 }]
                     ;[(and (= ace-backend "xh035-3V3") (= ace-id "op8"))
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op8"))
                      { "init" [3.0 6.0]
                        "min" [1.0 4.0 ]
                        "max" [4.0 8.0]
                        "grid" 1.0 }]
                     ;[(and (= ace-backend "xh018-1V8") (= ace-id "op8"))
                     ; { "init" [0.3 0.6]
                     ;   "min" [0.1 0.4 ]
                     ;   "max" [0.4 0.8]
                     ;   "grid" 0.1 }]
                     [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) (= ace-id "op9"))
                      { "init" [6.0 12.0 12.0 12.0 12.0 24.0]
                        "min" [1.0 1.0 1.0 1.0 1.0 3.0]
                        "max" [30.0 30.0 30.0 30.0 30.0 60.0]
                        "grid" 1.0 }]
                     [True (dfor (, k v) 
                                 (.items {"init" 3.0 "min" 1.0 "max" 30.0 "grid" 1.0})
                                 [k (cond [(= ace-id "op1") (np.repeat v 2) ]
                                          [(= ace-id "op2") (np.repeat v 2) ]
                                          [(= ace-id "op3") (np.repeat v 3) ]
                                          [(= ace-id "op4") (np.repeat v 3) ]
                                          [(= ace-id "op5") (np.repeat v 4) ]
                                          [(= ace-id "op6") (np.repeat v 2) ]
                                          [(= ace-id "op8") (np.repeat v 4) ]
                                          [(= ace-id "op9") (np.repeat v 6) ])]) ])
         "rc"  { "init" 4.0
                 "max"  15.0
                 "min"  1.0
                 "grid" 0.1 }
         "cc" { "init" 1.2
                "max"  12.0
                "min"  0.5
                "grid" 0.5 }
         #_/ })))

(defn op-to-args [^(of dict str float) op]
  """
  Converts ACE operating points parametes to args by replacing ':' with '_'.
  """
  (dfor (, p v) (.items op) [ (p.replace ":" "_") v]))

(defn input-parameters ^(of list str) [ace ^str ace-id ^int ace-variant]
  """
  Returns a list of input parameter names
  """
  (cond [(in ace-variant [1 3])
         (ac.sizing-identifiers ace)]
        [(and (= ace-id "op1") (in ace-variant [0 2])) 
          [ "MNCM1R:gmoverid" "MPCM2R:gmoverid" "MPCS1:gmoverid" "MND1A:gmoverid"
            "MNCM1R:fug"      "MPCM2R:fug"      "MPCS1:fug"      "MND1A:fug" 
            ;"res" "cap" 
            "MNCM1A:id" "MNCM1B:id" ]]
        [(and (= ace-id "op2") (in ace-variant [0 2])) 
          [ "MNCM11:gmoverid" "MPCM221:gmoverid" "MNCM31:gmoverid" "MND11:gmoverid"
            "MNCM11:fug"      "MPCM221:fug"      "MNCM31:fug"      "MND11:fug" 
            "MNCM12:id"       "MNCM32:id" ]]
        [(and (= ace-id "op3") (in ace-variant [0 2])) 
          [ "MNCM11:gmoverid" "MPCM221:gmoverid" "MNCM31:gmoverid" "MND11:gmoverid"
            "MNCM11:fug"      "MPCM221:fug"      "MNCM31:fug"      "MND11:fug" 
            "MNCM12:id"       "MNCM32:id" "MNCM31:id" ]]
        [(and (= ace-id "op4") (in ace-variant [0 2])) 
          [ "MNCM11:gmoverid" "MPCM221:gmoverid" "MNCM31:gmoverid" "MND11:gmoverid" 
            "MPC12:gmoverid"  "MPC1R:gmoverid" 
            "MNCM11:fug"      "MPCM221:fug"      "MNCM31:fug"      "MND11:fug"  
            "MPC12:fug"       "MPC1R:fug" 
            "MNCM13:id"       "MNCM32:id"        "MNCM31:id" ]]
        [(and (= ace-id "op5") (in ace-variant [0 2])) 
          [ "MNCM11:gmoverid" "MPCM221:gmoverid" "MNCM31:gmoverid" "MND11:gmoverid" 
            "MPC12:gmoverid"  "MPC1R:gmoverid" 
            "MNCM11:fug"      "MPCM221:fug"      "MNCM31:fug"      "MND11:fug"  
            "MPC12:fug"       "MPC1R:fug" 
            "MNCM13:id"       "MNCM32:id"        "MNCM31:id"       "MNCM12:id" ]]
        [(and (= ace-id "op6") (in ace-variant [0 2])) 
          [ "MNCM11:gmoverid" "MPCM21:gmoverid" "MPCS:gmoverid" "MND11:gmoverid" 
            "MPR1:gmoverid"   "MPC1:gmoverid"
            "MNCM11:fug"      "MPCM21:fug"      "MPCS:fug"      "MND11:fug"  
            "MPR1:fug"        "MPC1:fug"        "MNCM12:id"     "MNCM13:id" ]]
        [(and (= ace-id "op8") (in ace-variant [0 2])) 
          [ "MNCM51:gmoverid" "MPCM41:gmoverid" "MPCM31:gmoverid" "MNCM21:gmoverid" 
            "MNCM11:gmoverid" "MND11:gmoverid" 
            "MNCM51:fug"      "MPCM41:fug"      "MPCM31:fug"      "MNCM21:fug" 
            "MNCM11:fug"      "MND11:fug"
            "MNCM53:id"       "MNCM21:id" ]]
        [(and (= ace-id "op9") (in ace-variant [0 2])) 
          [ "MNCM41:gmoverid" "MPCM31:gmoverid" "MPCM21:gmoverid" "MNCM11:gmoverid" 
            "MND11:gmoverid"  "MNLS11:gmoverid" "MNR1:gmoverid"   "MPR2:gmoverid"
            "MNCM41:fug"      "MPCM31:fug"      "MPCM21:fug"      "MNCM11:fug" 
            "MND11:fug"       "MNLS11:fug"      "MNR1:fug"        "MPR2:fug"
            "MNCM43:id"       "MNCM44:id"       "MNCM42:id"       "MPCM32:id" 
            "MPCM33:id"       "MPCM34:id" ]]
        [True 
         (raise (NotImplementedError errno.ENOSYS
                                     (os.strerror errno.ENOSYS) 
                                     (.format "No parameters for {}-v{}."
                                     ace-id ace-variant)))]))
