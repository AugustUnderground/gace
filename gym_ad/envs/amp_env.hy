(import yaml)
(import [h5py :as h5])
(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])
(import [joblib :as jl])
(import [functools [partial]])
(import [operator [itemgetter]])
(import [datetime [datetime :as dt]])

(import gym)
(import [gym.spaces [Dict Box Discrete Tuple]])

(import [.prim_dev [*]])
(import [.util [*]])

(import [sphyctre [OpAnalyzer]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defclass AmplifierXH035Env [gym.Env]
  """
  Abstract parent class for all analog amplifier environments designed with
  the X-FAB XH035 Technology.
  """

  (setv metadata {"render.modes" ["human"]})

  (defn __init__ [self ^AmplifierID amp-id
                  ^str pdk-path ^str ckt-path ^str tech-cfg 
                  ^str nmos-path ^str pmos-path
                  ^int max-moves 
       &optional ^float [target-tolerance 1e-3] ^bool [close-target True] 
                 ^str   [data-log-prefix ""]
                 #_/ ] 
    """
    Initialzies the basics required by every amplifier implementing this
    interface.
    """

    (.__init__ (super AmplifierXH035Env self))

    ;; Logging the data means, a dataframe containing the sizing and
    ;; performance parameters will be written to an HDF5.
    ;; If no `data-log-prefix` is provided, the data will be discarded after each
    ;; episode.
    (setv self.data-log-prefix  data-log-prefix
          self.data-log       (pd.DataFrame))

    ;; Initialize parameters
    (setv self.last-reward    (- np.inf)
          self.max-moves      max-moves
          self.reset-counter  0)

    ;; Define list of universal performances for all Amplifiers
    (setv self.performance-parameters ["A0dB" "ugbw" "PM" "GM" "SR-r" "SR-f" 
                                    "vn-1Hz" "vn-10Hz" "vn-100Hz" "vn-1kHz" 
                                    "vn-10kHz" "vn-100kHz" 
                                    "psrr-n" "psrr-p" "cmrr" 
                                    "vo-lo" "vo-hi" "vi-lo" "vi-hi"
                                    "voff-stat" "voff-syst" 
                                    "i-out-max" 
                                    ;"i-out-min" 
                                    "A"])
    
    ;; This parameters specifies at which point the specification is considered
    ;; 'met' and the agent recieves its award.
    (setv self.target-tolerance target-tolerance)
    
    ;; If `True` the agent will be reset in a location close to the target.
    (setv self.close-target close-target)
                                    
    ;; Load the PyTorch NMOS/PMOS Models for converting paramters.
    (setv self.nmos (PrimitiveDevice f"{nmos-path}/model.pt" 
                                     f"{nmos-path}/scale.X" 
                                     f"{nmos-path}/scale.Y")
          self.pmos (PrimitiveDevice f"{pmos-path}/model.pt" 
                                     f"{pmos-path}/scale.X" 
                                     f"{pmos-path}/scale.Y"))

    ;; Amplifier ID:
    ;;  1 - Miller Operational Amplifier
    ;;  2 - Symmetrical Amplifier
    (setv self.amp-id amp-id)

    ;; Technology definitions such as min/max W/L, grid, etc.
    (setv self.tech-cfg (with [y (open tech-cfg)] (yaml.safe-load y)))

    ;; The `analyzer` communicates with the spectre simulator and returns the 
    ;; circuit performance.
    (setv self.ckt-path ckt-path
          self.pdk-path pdk-path
          self.analyzer (OpAnalyzer self.ckt-path self.pdk-path)))
  
  (defn render [self &optional ^str [mode "human"]]
    """
    Prints a generic ASCII Amplifier symbol for 'human' mode, in case the
    derived amplifier doesn't implement its own render method (which it
    should).
    """
    (let [ascii-amp (.format "
            VDD
             |         
          |\ |         
          | \|   Generic Amplifier Subcircuit
  INP ----+  + 
          |   \
          |    \
    B ----+ op  >---- O
          |    /
          |   /
  INN ----+  +
          | /|
          |/ |
             |
            VSS ") ]
      (cond [(= mode "human")
             (print ascii-amp)
             ascii-amp]
          [True 
           (raise (NotImplementedError f"Only 'human' mode is implemented."))])))

  (defn close [self]
    """
    Closes the spectre session.
    """
    (del self.analyzer)
    (setv self.analyzer None))

  (defn seed [self rng-seed]
    """
    Sets The RNG Seed for this environment.
    """
    (.seed np.random rng-seed)
    (.manual-seed pt rng-seed)
    rng-seed)

  (defn reset ^np.array [self]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.
    If `close-target` is true, an initial sizing will be found via bayesian
    optimization, placing the agent fairly close to the target.

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (unless self.analyzer
      (setv self.analyzer (OpAnalyzer self.ckt-path self.pdk-path)))

    ;; Reset the step counter and increase the reset counter.
    (setv self.moves         (int 0)
          self.reset-counter (inc self.reset-counter))

    ;; Clear the data log. If self.log-data == True the data will be written to
    ;; an HDF5 in the `done` function, otherwise it will be discarded.
    (setv self.data-log (pd.DataFrame))

    ;; Starting parameters are either random or close to a known solution.
    (setv parameters (self.starting-point :random (not self.close-target) 
                                          :noise True))
    
    ;; Target can be random or close to a known acheivable.
    (setv self.target (self.target-specification :noisy True))

    (setv self.performance (| (self.analyzer.simulate parameters)
                              {"A" (calculate-area self.amp-id parameters)}))

    (.observation self))

  (defn starting-point ^dict [self &optional ^bool [random False] 
                                             ^bool [noise True]]
    """
    Generate a starting point for the agent.
    Arguments:
      [random]:   Random starting point. (default = False)
      [noise]:    Add noise to found starting point. (default = True)
    Returns:      Starting point sizing.
    """
    (let [sizing (if random (random-sizing self.amp-id self.tech-cfg) 
                            (initial-sizing self.amp-id TechID.XH035))]
      (if noise
          (dfor (, p s) (.items sizing) 
                [p (if (or (.startswith p "W") (.startswith p "L")) 
                       (+ s (np.random.normal 0 1e-7)) s)])
          sizing)))

  (defn size-step ^tuple [self ^dict action]
    """
    Takes geometric parameters as dictionary and sets them in the netlist.
    This method is supposed to be called from a derived class, after converting
    electric parameters to geometric ones.

    Each circuit has to make sure the geometric parameters are within reason.
    (see `clip-sizing` mehtods.)
    """

    ;(setv self.performance (self.analyzer.evaluate-circuit action)

    (setv self.performance (| (self.analyzer.simulate action)
                              {"A" (calculate-area self.amp-id action)})
          self.data-log 
          (self.data-log.append (dfor (, k v) (.items self.performance) [k v])
                                      ;[k (first v)])
                                :ignore-index True))

    (, (.observation self) (.reward self) (.done self) (.info self)))
 
  (defn observation ^np.array [self]
    """
    Returns a 'observation-space' conform dictionary with the current state of
    the circuit and its performance.
    """
    (let [p-getter (itemgetter #* self.performance-parameters)
        s-getter (itemgetter #* (-> self.performance (.keys) (set) 
                                    (.difference self.performance-parameters) 
                                    (list)))
          perf (->> self.performance (p-getter) (flatten) (np.array))
          targ (->> self.target (p-getter) (map np.mean) (list) (np.array))

          dist (np.array (list (map (fn [f p] (f (np.nan-to-num p)))
                                    (p-getter (self.individual-rewards)) perf)))

          stat (-> self.performance (s-getter) (flatten) (np.array))

          obs (-> (, perf targ dist stat) 
                  (np.hstack) 
                  (np.squeeze) 
                  (np.float32))]

      (np.nan-to-num obs)))

  (defn individual-rewards ^dict [self]
    """
    Hand crafted reward functions for each individual performance parameter.
    """
    {"A0dB"      (absolute-condition (. self.target ["A0dB"])               '<=)  ; t dB ≤ x
     "ugbw"      (ranged-condition #* (. self.target ["ugbw"]))                   ; t1 Hz ≤ x & t2 Hz ≥ x
     "PM"        (absolute-condition (. self.target ["PM"])                 '<=)  ; t dB ≤ x
     "GM"        (absolute-condition (np.abs (. self.target ["GM"]))        '<=)  ; t ° ≤ |x|
     "SR-r"      (ranged-condition #* (. self.target ["SR-r"]))                   ; t1 V/s ≤ x & t2 V/s ≥ x
     "SR-f"      (ranged-condition #* (np.abs (. self.target ["SR-f"])))          ; t1 V/s ≤ |x| & t2 V/s ≥ x
     "vn-1Hz"    (absolute-condition (. self.target ["vn-1Hz"])             '>=)  ; t V ≥ x
     "vn-10Hz"   (absolute-condition (. self.target ["vn-10Hz"])            '>=)  ; t V ≥ x
     "vn-100Hz"  (absolute-condition (. self.target ["vn-100Hz"])           '>=)  ; t V ≥ x
     "vn-1kHz"   (absolute-condition (. self.target ["vn-1kHz"])            '>=)  ; t V ≥ x
     "vn-10kHz"  (absolute-condition (. self.target ["vn-10kHz"])           '>=)  ; t V ≥ x
     "vn-100kHz" (absolute-condition (. self.target ["vn-100kHz"])          '>=)  ; t V ≥ x
     "psrr-n"    (absolute-condition (. self.target ["psrr-n"])             '<=)  ; t dB ≤ x
     "psrr-p"    (absolute-condition (. self.target ["psrr-p"])             '<=)  ; t dB ≤ x
     "cmrr"      (absolute-condition (. self.target ["cmrr"])               '<=)  ; t dB ≤ x
     "vi-lo"     (absolute-condition (. self.target ["vi-lo"])              '>=)  ; t V ≥ x
     "vi-hi"     (absolute-condition (. self.target ["vi-hi"])              '<=)  ; t V ≤ x
     "vo-lo"     (absolute-condition (. self.target ["vo-lo"])              '>=)  ; t V ≥ x
     "vo-hi"     (absolute-condition (. self.target ["vo-hi"])              '<=)  ; t V ≤ x
     ;"i-out-min" (absolute-condition (. self.target ["i-out-min"])          '<=)  ; t A ≤ x
     "i-out-max" (absolute-condition (. self.target ["i-out-max"])          '>=)  ; t A ≥ x
     "voff-stat" (absolute-condition (. self.target ["voff-stat"])          '>=)  ; t V ≥ x
     "voff-syst" (absolute-condition (np.abs (. self.target ["voff-syst"])) '>=)  ; t V ≥ |x|
     "A"         (absolute-condition (. self.target ["A"])                  '>=)  ; t μm^2 ≥ x
     #_/ })

  (defn reward ^float [self &optional ^dict [performance {}]
                                      ^list [params []]]
    """
    Calculates a reward based on the target and the current perforamnces.
    Arguments:
      [performance]:  Dictionary with performances.
      [target]:       Dictionary with target values.
      [params]:       List of parameters.
      
      **NOTE**: Both dictionaries must include the keys defined in `params`.
    If no arguments are provided, the current state of the object is used to
    calculate the reward.
    """
    (let [perf-dict  (or performance self.performance) 
          p-getter (itemgetter #* self.performance-parameters)
          params     (or params self.performance-parameters)
          reward-fns (.individual-rewards self)
          rewards (lfor p params 
                     ((. reward-fns [p]) 
                      (np.nan-to-num (. perf-dict [p])))) ]
      (-> rewards (np.array) (np.sum) (np.abs) (np.log10) (-) (float))))
 
  (defn done ^bool [self]
    """
    Returns True if the target is met (under consideration of the
    'target-tolerance'), or if moves > max-moves, otherwise False is returned.
    """
    (let [perf (np.array (list (map #%(->> %1 (get self.performance) (np.nan-to-num) (np.mean))
                                    self.performance-parameters)))
          targ (np.array (list (map #%(->> %1 (get self.target) (np.nan-to-num) (np.mean))
                                    self.performance-parameters)))
          loss (Loss.MAE perf targ)
          time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))]

      ;; If a log path is defined, a HDF5 data log is kept with all the sizing
      ;; parameters and corresponding performances.
      ;(when self.data-log-prefix
      ;  (setv log-file (.format "{}-{}.h5" self.data-log-prefix time-stamp))
      ;  (with [h5-file (h5.File log-file "w")]
      ;   (for [col self.data-log.columns]
      ;     (setv (get h5-file (-> col (.replace ":" "-") (.replace "." "_"))) 
      ;        (.to-numpy (get self.data-log col))))))

      ;; 'done' when either maximum number of steps are exceeded, or the
      ;; overall loss is less than the specified target loss.
      (or (> self.moves self.max-moves) 
          (< loss self.target-tolerance))))

  (defn info ^dict [self]
    """
    Returns very useful information about the current state of the circuit,
    simulator and live in general.
    """
    {"observation-key" (+ (list (sum (zip #* (lfor pp self.performance-parameters 
                                                   (, f"performance_{pp}"
                                                      f"target_{pp}"
                                                      f"distance_{pp}"))) 
                                     (,)))
                          (lfor sp (.keys self.performance) 
                                   :if (not-in sp self.performance-parameters) 
                               sp))
     #_ /}))
