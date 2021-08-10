(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [fractions [Fraction]])
(import [decimal [Decimal]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])
(import [joblib :as jl])

(import gym)
(import [gym.spaces [Dict Box Discrete Tuple]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defn map2dict [java-map]
  """
  Convert a java Map to a python dict. 
  **DOESN'T WORK FOR NESTED MAPS!**
  Arguments:
    java-map: java Map
  Returns:    dict
  """
  (dfor k (-> java-map (.keySet) (.toArray))
    [k (if (isinstance (setx w (.get java-map k)) Iterable)
        (np.array (list w)) w)]))

(defn scale-value ^float [^float x ^float x-min ^float x-max]
  """
  Scale a value between [0;1]
    x′ = (x - x_min) ÷ (x_max - x_min)
  """
  (/ (- x x-min) (- x-max x-min)))

(defn unscale-value ^float [^float x′ ^float x-min ^float x-max]
  """
  Scales a value ∈ [0;1] back to its original.
    x = (x_max - x_min) · x′ + x_min
  """
  (+ (* (- x-max x-min) x′) x-min))

(defn dec2frac ^tuple [^float ratio]
  """
  Turns a float decimal into an integer fraction.
  """
  (as-> ratio it (str it) (Decimal it) (Fraction it) 
                 (, it.numerator it.denominator)))

(defn frac2dec ^float [^int num ^int den]
  """
  Turns a fraction into a float ratio.
  """
  (/ num den))

(defclass AmplifierEnv [gym.Env]
  """
  Abstract parent class for all analog amplifier environments.
  """

  (setv metadata {"render.modes" ["human"]})

  (defn __init__ [self amplifier lib-path sim-path ckt-path
                       max-moves target-tolerance ]
    """
    Initialzies the basics required by every amplifier implementing this
    interface.
    """

    (.__init__ (super AmplifierEnv self))

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
    
    ;; The `Dict` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv perf (dfor pp self.performance-parameters
                  [f"{pp}" (Box :low (- np.inf) :high np.inf 
                                :shape (, 1) :dtype np.float32)])
          targ (dfor pp self.performance-parameters
                  [f"t_{pp}" (Box :low (- np.inf) :high np.inf 
                                  :shape (, 1) :dtype np.float32)])
          dist (dfor pp self.performance-parameters
                  [f"∆_{pp}" (Box :low (- np.inf) :high np.inf 
                                  :shape (, 1) :dtype np.float32)])
          wavs {"loopGainAbs" (Box :low (- np.inf) :high np.inf 
                                   :shape (501,) :dtype np.float32)
                "loopGainPhase" (Box :low (- np.inf) :high np.inf 
                                     :shape (501,) :dtype np.float32)
                "cmrr" (Box :low (- np.inf) :high np.inf 
                            :shape (601,) :dtype np.float32)
                "psrr_p" (Box :low (- np.inf) :high np.inf 
                              :shape (601,) :dtype np.float32)
                "psrr_n" (Box :low (- np.inf) :high np.inf 
                              :shape (601,) :dtype np.float32) }
          ;self.observation-space (Dict { #** perf #** targ #** dist #** wavs}))
          self.observation-space (Dict (| perf targ dist wavs)))
                                     
    ;; The amplifier object `op` communicates through java with spectre and
    ;; returns performances and other simulation / analyses results.
    (setv self.lib-path lib-path
          self.sim-path sim-path
          self.ckt-path ckt-path)

    (setv self.op None))
    ;; Reset should take care of this?
    ;(setv self.op (amplifier.build self.sim-path self.lib-path self.ckt-path))
    ;(.start self.op)
  
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
    (del self.op)
    (setv self.op None))

  (defn reset [self]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.
    If `close-targe` is true, an initial sizing will be found via bayesian
    optimization, placing the agent fairly close to the target.

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (unless self.op
      (setv self.op (amplifier.build self.sim-path self.lib-path self.ckt-path))
      (.start self.op))

    (setv self.moves 0)
    (setv self.reset-counter (inc self.reset-counter))

    (unless self.same-target
      (setv self.target (self.random-target)))

    (setv parameters 
          (if self.close-target
              ; TODO FIXME IMPLEMENT CLOSE TARGET
            (self.random-parameters)
            (self.random-parameters)))

    (for [(, param value) parameters]
      (self.op.set param value))

    (self.simulate)
    (self.observation))

  (defn random-target ^dict [self]
    """
    Generate a random target specification.
    """
    (dfor pp self.performance-parameters
      ; TODO FIXME RETURN ACHIEVABLE TARGET PERFORAMNCE
      [pp (np.random.rand)]))

  (defn random-parameters [self]
    """
    Generate random Sizing parameters.
    """
    (-> self.op (.getRandomValues) (map2dict)))

  (defn simulate [self]
    """
    Run a simulation with the current parameters and update the performances.
    """
    (.simulate self.op)
    (setv self.performance (map2dict (.getPerformanceValues self.op))
          self.waveforms   (map2dict (.getWaves self.op)))
    self.performance)

  (defn step ^tuple [self ^dict action]
    """
    Takes geometric parameters as dictionary and sets them in the netlist.
    This method is supposed to be called from a derived class, after converting
    electric parameters to geometric ones.
    """
    (for [(, param value) (.items action)]
      (self.op.set param value))
      (self.simulate)
      (, (.observation self) (.reward self) (.done self) (.information self)))
 
  (defn observation [self]
    """
    Returns a 'observation-space' conform dictionary with the current state of
    the circuit and its performance.
    """
    (let [dist (dfor p self.performance-parameters
                   [(.format "∆_{}" p) 
                    (np.abs (- (get self.performance p) 
                               (get self.target p)))])
          targ (dfor (, p t) (.items self.target) [(.format "t_{p}") t]) ]
      (| self.performance targ dist self.waveforms)))

  (defn reward ^float [self &optional ^dict [performance self.perforamnce]
                                      ^dict [target self.target]]
    """
    Calculates a reward based on the target and the current perforamnces.

    TODO: Try different reward / cost functions.
    """
    (- (np.sum) (lfor p (.keys performance)
                      (* (np.abs (/ (- (get performance p) (get target p)) 
                                    (get target p))) 100))))
 
  (defn done ^bool [self]
    """
    Returns True if the target is met (under consideration of the
    'target-tolerance'), or if moves > max-moves, otherwise False is returned.
    """
    (or (> self.moves self.max-moves) 
        (<= (np.abs (.reward self)) self.target-tolerance)))

  (defn info [self]
    """
    Returns very useful information about the current state of the circuit,
    simulator and live in general.
    """
    ; TODO FIXME: SHOW USEFUL INFORMATION
    {"nothing" None
     "to" 666
     "see" "here"}))
