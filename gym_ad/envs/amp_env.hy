(import [itertools [product]])

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
  Arguments:
    java-map: java Map
  Returns:    dict
  """
  (dfor k (-> java-map (.keySet) (.toArray))
    [k (if (in (type (setx w (.get java-map k))) [float int])
        w (np.array (list w)))]))

(defn scale-value ^float [^float x ^float x-min ^float x-max]
  """
  Scale a value between [0;1]
  """
  (/ (- x x-min) (- x-max x-min)))

(defclass AmplifierEnv [gym.Env]
  (setv metadata {"render.modes" ["human"]})

  (defn __init__ [self amplifier lib-path sim-path ckt-path
                       max-moves target-tolerance ]

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
    (setv self.op (amplifier.build self.sim-path self.lib-path self.ckt-path))
    (.start self.op))
  
  (defn random-parameters [self]
    """
    Generate random Sizing parameters.
    """
    (-> self.op (.getRandomValues) (map2dict)))

  (defn render [self &optional [mode "human"]]
    ; TODO FIXME Amplifier symbol
    (print "Generic amplifier.")
    None)

  (defn close [self]
    ; TODO FIXME DESTROY / CLOSE SPECTRE SESSION
    None)

  (defn reset [self]
    (setv self.moves 0)
    (setv self.reset-counter (inc self.reset-counter))

    (unless self.same-target
      (setv self.target (self.random-target)))

    (setv parameters 
          (if self.close-target
            (self.random-parameters) 
            (self.random-parameters)))

    (for [(, param value) parameters]
      (self.op.set param value))

    (self.simulate)
    (self.observe))

  (defn random-target ^dict [self]
    (dfor pp self.performance-parameters
      ; TODO FIXME RETURN ACHIEVABLE TARGET PERFORAMNCE
      [pp (np.random.rand)]))

  (defn simulate [self]
    (.simulate self.op)
    (setv self.performance (map2dict (.getPerformanceValues self.op))
          self.waveforms   (map2dict (.getWaves self.op))))

  (defn step [self ^dict action]
    (for [(, param value) (.items action)]
      (self.op.set param value))
      (self.simulate)
      (, (.observation self) (.reward self) (.done self) (.information self)))
 
  (defn observation [self]
    (let [dist (dfor p self.performance-parameters
                   [(.format "∆_{}" p) 
                    (np.abs (- (get self.performance p) 
                               (get self.target p)))])
          targ (dfor (, p t) (.items self.target) [(.format "t_{p}") t]) ]
      (| self.performance targ dist self.waveforms)))

  (defn reward ^float [self &optional ^dict [performance self.perforamnce]]
      ; TODO FIXME RETURN ACTUAL REWARD
    (np.random.rand))
 
  (defn done ^bool [self &optional ^int   [moves self.moves] 
                                       ^float [reward self.reward]]
    ; TODO FIXME RETURN ACTUAL DONE FLAG
    (np.random.choice [True False]))

  (defn info [self]
    ; TODO FIXME SEND USEFUL INFORMATION
    {"nothing" None
     "to" 666
     "see" "here"}))
