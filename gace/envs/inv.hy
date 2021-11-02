(import yaml)
(import json)
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

(import [.util.prim [*]])
(import [.util.util [*]])

(import [hace :as ac])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defclass Nand4Env [gym.Env]
  """
  Abstract parent class for all inverter environments.
  """

  (setv metadata {"render.modes" ["human"]})

  (defn __init__ [self ^(of list str) pdk-path ^str ckt-path
                  ^int max-moves ^bool random-start 
                  ^float tolerance 
       &optional ^str   [data-log-prefix ""]
                 #_/ ] 
    """
    Initialzies the basics required by every inverter implementing this
    interface.
    """

    (.__init__ (super Nand4Env self))

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
    (setv self.performance-parameters ["vs0" "vs2" "vs1" "vs3"])
  
    ;; Loss function: | performance - target | / target
    (setv self.tolerance tolerance
          self.loss (fn [x y] (/ (np.abs (- x y)) y)))

    ;; Whether to use a random starting value or the initial ones.
    (setv self.random-start random-start)

    ;; The `inverter` communicates with the spectre simulator and returns the 
    ;; circuit performance.
    (setv self.ckt-path ckt-path
          self.pdk-path pdk-path
          self.inverter (ac.nand-4 self.ckt-path :pdk-path self.pdk-path))

    ;; Specify constants as they are defined in the netlist and by the PDK.
    (setv params    (ac.current-parameters self.inverter)
          self.vdd  (get params "vdd")                ; Supply voltage 
          #_/ ))

  (defn render [self &optional ^str [mode "human"]]
    """
    Prints a generic ASCII Amplifier symbol for 'human' mode, in case the
    derived inverter doesn't implement its own render method (which it
    should).
    """
    (let [ascii-inv (.format "
      +-----+
      |  1  |
 A ---|     |O--- Q
      |     |
      +-----+
      ") ]
      (cond [(= mode "human")
             (print ascii-inv)]
          [True 
           (raise (NotImplementedError f"Only 'human' mode is implemented."))])))
  (defn close [self]
    """
    Closes the spectre session.
    """
    (.stop self.inverter)
    (del self.inverter)
    (setv self.inverter None))

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

    (unless self.inverter
      (setv self.inverter (ac.nand-4 self.ckt-path :pdk-path self.pdk-path)))

    ;; Reset the step counter and increase the reset counter.
    (setv self.moves         (int 0)
          self.reset-counter (inc self.reset-counter))

    ;; Clear the data log. If self.log-data == True the data will be written to
    ;; an HDF5 in the `done` function, otherwise it will be discarded.
    (setv self.data-log (pd.DataFrame))

    ;; Starting parameters are either random or close to a known solution.
    (setv parameters (self.starting-point :random self.random-start
                                          :noise (not self.random-start)))
    
    ;; Target is always VDD/2 for every switching voltage
    (setv self.target (dict (zip self.performance-parameters 
                                 (repeat (/ self.vdd 2.0) 
                                         (len self.performance-parameters)))))

    ;; Get the current performance for the initial parameters
    (setv self.performance (ac.evaluate-circuit self.inverter :params parameters))

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
    (let [sizing (if random (ac.random-sizing self.inverter) 
                            (ac.initial-sizing self.inverter))]
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
    """

    (setv self.performance (ac.evaluate-circuit self.inverter :params action)
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
          perf (->> self.performance (p-getter) (np.array))
          targ (->> self.target (p-getter) (np.array))
          dist (self.loss perf targ)

          obs (-> (, perf targ dist) 
                  (np.hstack) 
                  (np.squeeze) 
                  (np.float32))]

      (np.nan-to-num obs)))

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
    (let [perf-dict    (or performance self.performance) 
          p-getter     (itemgetter #* self.performance-parameters)
          params       (or params self.performance-parameters)

          performances (-> perf-dict (p-getter) (np.array))
          targets      (-> self.target (p-getter) (np.array))
          cost         (/ (np.abs (- performances targets)) targets) ]

       (when (or (np.any (np.isnan cost)) (np.any (np.isinf cost)))
          (let [time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
              json-file (.format "./parameters-{}.json" time-stamp) ]
            (ac.dump-state self.amplifier :file-name json-file)))

       (-> cost (np.nan-to-num) (np.sum) (np.float32) (-))))

  (defn done ^bool [self]
    """
    Returns True if the target is met (all predicatets given in 
    """
    (let [p-getter (itemgetter #* self.performance-parameters)
          perf (list (map #%(->> %1 (get self.performance) (np.nan-to-num) (np.mean))
                                    self.performance-parameters))
          targ (list (map #%(->> %1 (get self.target) (np.nan-to-num) (np.mean))
                                    self.performance-parameters))

          loss (np.sum (self.loss (np.array targ) (np.array perf)))
          
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
      (bool (or (> self.moves self.max-moves) (<= loss self.tolerance)))))

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
     #_/ }))
