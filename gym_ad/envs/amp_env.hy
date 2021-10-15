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

(import [hace :as ac])

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
                  ^str pdk-path ^str ckt-path
                  ^str nmos-path ^str pmos-path
                  ^int max-moves 
       &optional ^float [target-tolerance 1e-3]
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
    (setv self.performance-parameters ["a_0" "ugbw" "pm" "gm" "sr_r" "sr_f" 
                                    "vn_1Hz" "vn_10Hz" "vn_100Hz" "vn_1kHz" 
                                    "vn_10kHz" "vn_100kHz" 
                                    "psrr_n" "psrr_p" "cmrr" 
                                    "v_ol" "v_oh" "v_il" "v_ih"
                                    "voff_stat" "voff_sys" 
                                    "overshoot_r" "overshoot_f"
                                    "i_out_max" 
                                    "i_out_min" "A"
                                    ])
    
    ;; This parameters specifies at which point the specification is considered
    ;; 'met' and the agent recieves its award.
    (setv self.target-tolerance target-tolerance)
    
    ;; If `True` the agent will be reset in a location close to the target.
    ;(setv self.close-target close-target)
                                    
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
    (setv self.amp-id amp-id
          self.amp-maker (cond [(= amp-id AmplifierID.SYMMETRICAL)
                                ac.single-ended-opamp]
                               [(= amp-id AmplifierID.MILLER)
                                ;ac.single-ended-opamp
                                (raise (NotImplementedError 
                                        f"Miller OP Not yet implemented."))]
                               [True (raise (NotImplementedError 
                                        f"Miller OP Not yet implemented."))]))

    ;; The `amplifier` communicates with the spectre simulator and returns the 
    ;; circuit performance.
    (setv self.ckt-path ckt-path
          self.pdk-path pdk-path
          self.amplifier (self.amp-maker self.ckt-path :pdk-path self.pdk-path)))
  
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
    (.stop self.amplifier)
    (del self.amplifier)
    (setv self.amplifier None))

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

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (unless self.amplifier
      (setv self.amplifier (self.amp-maker self.ckt-path :pdk-path self.pdk-path)))

    ;; Reset the step counter and increase the reset counter.
    (setv self.moves         (int 0)
          self.reset-counter (inc self.reset-counter))

    ;; Clear the data log. If self.log-data == True the data will be written to
    ;; an HDF5 in the `done` function, otherwise it will be discarded.
    (setv self.data-log (pd.DataFrame))

    ;; Starting parameters are either random or close to a known solution.
    (setv parameters (self.starting-point :random self.random-target
                                          :noise (not self.random-target)))
    
    ;; Target can be random or close to a known acheivable.
    (setv self.target (self.target-specification :random self.random-target 
                                                 :noisy True))

    ;; Get the current performance for the initial parameters
    (setv self.performance (ac.evaluate-circuit self.amplifier :params parameters))

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
    (let [sizing (if random (ac.random-sizing self.amplifier) 
                            (ac.initial-sizing self.amplifier))]
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

    (setv self.performance (ac.evaluate-circuit self.amplifier :params action)
          self.data-log 
          (self.data-log.append (dfor (, k v) (.items self.performance) [k v])
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
    {"a_0"         (absolute-condition (. self.target ["a_0"])                '<=)  ; t dB ≤ x
     "ugbw"        (ranged-condition #* (. self.target ["ugbw"]))                   ; t1 Hz ≤ x & t2 Hz ≥ x
     "pm"          (absolute-condition (. self.target ["pm"])                 '<=)  ; t dB ≤ x
     "gm"          (absolute-condition (np.abs (. self.target ["gm"]))        '<=)  ; t ° ≤ |x|
     "sr_r"        (ranged-condition #* (. self.target ["sr_r"]))                   ; t1 V/s ≤ x & t2 V/s ≥ x
     "sr_f"        (ranged-condition #* (np.abs (. self.target ["sr_f"])))          ; t1 V/s ≤ |x| & t2 V/s ≥ x
     "vn_1Hz"      (absolute-condition (. self.target ["vn_1Hz"])             '>=)  ; t V ≥ x
     "vn_10Hz"     (absolute-condition (. self.target ["vn_10Hz"])            '>=)  ; t V ≥ x
     "vn_100Hz"    (absolute-condition (. self.target ["vn_100Hz"])           '>=)  ; t V ≥ x
     "vn_1kHz"     (absolute-condition (. self.target ["vn_1kHz"])            '>=)  ; t V ≥ x
     "vn_10kHz"    (absolute-condition (. self.target ["vn_10kHz"])           '>=)  ; t V ≥ x
     "vn_100kHz"   (absolute-condition (. self.target ["vn_100kHz"])          '>=)  ; t V ≥ x
     "psrr_n"      (absolute-condition (. self.target ["psrr_n"])             '<=)  ; t dB ≤ x
     "psrr_p"      (absolute-condition (. self.target ["psrr_p"])             '<=)  ; t dB ≤ x
     "cmrr"        (absolute-condition (. self.target ["cmrr"])               '<=)  ; t dB ≤ x
     "v_il"        (absolute-condition (. self.target ["v_il"])               '>=)  ; t V ≥ x
     "v_ih"        (absolute-condition (. self.target ["v_ih"])               '<=)  ; t V ≤ x
     "v_ol"        (absolute-condition (. self.target ["v_ol"])               '>=)  ; t V ≥ x
     "v_oh"        (absolute-condition (. self.target ["v_oh"])               '<=)  ; t V ≤ x
     "i_out_min"   (absolute-condition (. self.target ["i-out-min"])          '<=)  ; t A ≤ x
     "i_out_max"   (absolute-condition (. self.target ["i_out_max"])          '>=)  ; t A ≥ x
     "overshoot_r" (absolute-condition (. self.target ["overshoot_r"])        '>=)  ; t A ≥ x
     "overshoot_f" (absolute-condition (. self.target ["overshoot_f"])        '>=)  ; t A ≥ x
     "voff_stat"   (absolute-condition (. self.target ["voff_stat"])          '>=)  ; t V ≥ x
     "voff_sys"    (absolute-condition (np.abs (. self.target ["voff_sys"]))  '>=)  ; t V ≥ |x|
     "A"           (absolute-condition (. self.target ["A"])                  '>=)  ; t μm^2 ≥ x
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
          p-getter   (itemgetter #* self.performance-parameters)
          params     (or params self.performance-parameters)
          reward-fns (.individual-rewards self)
                      ;((. reward-fns [p]) (np.nan-to-num (. perf-dict [p])))
          rewards    (lfor p params 
                      (let [pp (. perf-dict [p])
                          rr (. reward-fns [p])]
                        (if (or (np.isnan pp) (np.isinf pp))
                          (-> -Inf (np.nan-to-num) (float))
                          (rr pp))))]
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
