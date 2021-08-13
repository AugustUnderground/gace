(import [h5py :as h5])
(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])
(import [joblib :as jl])

(import gym)
(import [gym.spaces [Dict Box Discrete Tuple]])

(import [.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defclass AmplifierEnv [gym.Env]
  """
  Abstract parent class for all analog amplifier environments.
  """

  (setv metadata {"render.modes" ["human"]})

  (defn __init__ [self amplifier ^str sim-path ^str pdk-path ^str ckt-path 
                  ^int max-moves 
       &optional ^float [target-tolerance 1e-3] ^bool [close-target True] 
                 ^bool  [data-log-path ""]] 
    """
    Initialzies the basics required by every amplifier implementing this
    interface.
    """

    (.__init__ (super AmplifierEnv self))

    ;; Logging the data means, a dataframe containing the sizing and
    ;; performance parameters will be written to an HDF5.
    ;; If no `data-log-path` is provided, the data will be discarded after each
    ;; episode.
    (setv self.data-log-path  data-log-path
          self.data-log       (pd.DataFrame))

    ;; Initialize parameters
    (setv self.last-reward    (- np.inf)
          self.max-moves      max-moves
          self.reset-counter  0)

    ;; Define list of universal performances for all Amplifiers
    (setv self.performance-parameters ["a_0" "ugbw" "pm" "gm" "sr_r" "sr_f"
                                       "vn_1Hz" "vn_10Hz" "vn_100Hz" "vn_1kHz"
                                       "vn_10kHz" "vn_100kHz" "psrr_p"
                                       "psrr_n" "cmrr" "v_il" "v_ih" "v_ol"
                                       "v_oh" "i_out_min" "i_out_max"
                                       "voff_stat" "voff_sys" "A"])
    
    ;; This parameters specifies at which point the specification is considered
    ;; 'met' and the agent recieves its award.
    (setv self.target-tolerance target-tolerance)
    
    ;; If `True` the agent will be reset in a location close to the target.
    (setv self.close-target close-target)

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf 
                                      :shape (, 202) :dtype np.float32))
                                     
    ;; The amplifier object `op` communicates through java with spectre and
    ;; returns performances and other simulation / analyses results.
    (setv self.sim-path sim-path
          self.pdk-path pdk-path
          self.ckt-path ckt-path)

    (setv self.op None
          self.amplifier amplifier))
  
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
    (.stop self.op)
    (del self.op)
    (setv self.op None))

  (defn reset ^np.array [self]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.
    If `close-target` is true, an initial sizing will be found via bayesian
    optimization, placing the agent fairly close to the target.

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (unless self.op
      (setv self.op (self.amplifier.build self.sim-path 
                                          self.pdk-path 
                                          self.ckt-path))
      (.start self.op))

    ;; Reset the step counter and increase the reset counter.
    (setv self.moves         (int 0)
          self.reset-counter (inc self.reset-counter))

    ;; Clear the data log. If self.log-data == True the data will be written to
    ;; an HDF5 in the `done` function, otherwise it will be discarded.
    (setv self.data-log (pd.DataFrame))

    ;; Starting parameters are either random or close to a known solution.
    (setv parameters (if self.close-target
                        (self.starting-point :random False :noise True)
                        (self.random-parameters :random True :noise False)))

    (for [(, param value) (.items parameters)]
      (self.op.set (str param) (np.float64 value)))

    ;; Target can be random or close to a known acheivable.
    (setv self.target (self.target-specification :noisy True))

    (.simulate self)
    (.observation self))

  (defn target-specification ^dict [self &optional [noisy True]]
    """
    Generate a noisy target specification.
    """
    {"a_0"       (+ 50.0       (if noisy (np.random.normal 0 5.0) 0))
     "ugbw"      (+ 3000000.0  (if noisy (np.random.normal 0 5e6) 0))
     "pm"        (+ 65.0       (if noisy (np.random.normal 0 3.0) 0))
     "gm"        (+ -30.0      (if noisy (np.random.normal 0 2.5) 0))
     "sr_r"      (+ 4000000.0  (if noisy (np.random.normal 0 5e6) 0))
     "sr_f"      (+ -4000000.0 (if noisy (np.random.normal 0 5e6) 0))
     "vn_1Hz"    (+ 5e-06      (if noisy (np.random.normal 0 5e-7) 0))
     "vn_10Hz"   (+ 2e-06      (if noisy (np.random.normal 0 5e-7) 0))
     "vn_100Hz"  (+ 5e-07      (if noisy (np.random.normal 0 5e-8) 0))
     "vn_1kHz"   (+ 1.5e-07    (if noisy (np.random.normal 0 5e-8) 0))
     "vn_10kHz"  (+ 6e-08      (if noisy (np.random.normal 0 5e-9) 0))
     "vn_100kHz" (+ 4e-08      (if noisy (np.random.normal 0 5e-9) 0))
     "psrr_p"    (+ 90.0       (if noisy (np.random.normal 0 5.0) 0))
     "psrr_n"    (+ 50.0       (if noisy (np.random.normal 0 5.0) 0))
     "cmrr"      (+ 100        (if noisy (np.random.normal 0 10.0) 0))
     "v_il"      (+ 0.5        (if noisy (np.random.normal 0 5e-2) 0))
     "v_ih"      (+ 3.0        (if noisy (np.random.normal 0 5e-1) 0))
     "v_ol"      (+ 1.5        (if noisy (np.random.normal 0 5e-2) 0))
     "v_oh"      (+ 1.5        (if noisy (np.random.normal 0 5e-2) 0))
     "i_out_min" (+ -2.5       (if noisy (np.random.normal 0 5e-2) 0))
     "i_out_max" (+ 2.5        (if noisy (np.random.normal 0 5e-2) 0))
     "voff_stat" (+ 3e-3       (if noisy (np.random.normal 0 5e-4) 0))
     "voff_sys"  (+ -1.5e-3    (if noisy (np.random.normal 0 5e-4) 0))
     "A"         (+ 5e-10      (if noisy (np.random.normal 0 5e-11) 0))})

  (defn starting-point ^dict [self &optional ^bool [random False] 
                                             ^bool [noise True]]
    """
    Generate a starting point for the agent.
    Arguments:
      [random]:   Random starting point. (default = False)
      [noise]:    Add noise to found starting point. (default = True)
    Returns:      Starting point sizing.
    """
    (let [sizing (map2dict (if random (.getRandomValues self.op)
                                      (.getInitValues self.op)))]
      (if noise
        (dfor (, p s) (.items sizing)
          [p (+ s (np.random.normal 0 1e-7))])
        sizing)))

  (defn simulate ^dict [self]
    """
    Run a simulation with the current parameters and update the performances.

    TODO: Extract waveforms and add to observation space.
    """
    (.simulate self.op)
    (setv self.performance (map2dict (.getPerformanceValues self.op)))
    self.performance)

  (defn step ^tuple [self ^dict action]
    """
    Takes geometric parameters as dictionary and sets them in the netlist.
    This method is supposed to be called from a derived class, after converting
    electric parameters to geometric ones.
    """
    (for [(, param value) (.items action)]
      (self.op.set param value))

    (setv self.data-log (self.data-log.append (self.simulate) 
                                              :ignore-index True))
    
    (, (.observation self) (.reward self) (.done self) (.info self)))
 
  (defn observation ^np.array [self]
    """
    Returns a 'observation-space' conform dictionary with the current state of
    the circuit and its performance.
    """
    (let [(, perf targ) (np.array (list (zip #* (lfor pp self.performance-parameters 
                                                      [ (get self.performance pp)
                                                         (get self.target pp)]))))

          dist (np.abs (- perf targ))

          stat (np.array (lfor sp (.keys self.performance) 
                                  :if (not-in sp self.performance-parameters) 
                               (get self.performance sp)))
          obs (-> (, perf targ dist stat) 
                  (np.hstack) 
                  (np.squeeze) 
                  (np.float32))]
      (np.where (np.isnan obs) (- np.inf) obs)))

  (defn reward ^float [self &optional ^dict [performance {}]
                                      ^dict [target {}]
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
    (let [perf-dict (or performance self.performance) 
          targ-dict (or target self.target)
          params    (or params self.performance-parameters)
          perf      (np.array (list (map perf-dict.get params)))
          targ      (np.array (list (map targ-dict.get params)))]
        (- (self.loss perf targ))))
 
  (defn done ^bool [self]
    """
    Returns True if the target is met (under consideration of the
    'target-tolerance'), or if moves > max-moves, otherwise False is returned.
    """
    (let [perf (np.array (list (map self.performance.get 
                                    self.performance-parameters)))
          targ (np.array (list (map self.target.get 
                                    self.performance-parameters)))
          loss (loss.MAE perf targ)]

      ;; If a log path is defined, a HDF5 data log is kept with all the sizing
      ;; parameters and corresponding performances.
      (when self.data-log-path
        (with [h5-file (h5.File self.data-log-path "a")]
          (for [col self.data-log]
            (setv (get h5-file col) (.to-numpy (get self.data-log col))))))

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
