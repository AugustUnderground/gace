(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import gym)
(import [gym.spaces [Dict Box Discrete MultiDiscrete Tuple]])

(import [gace.util.func [*]])
(import [gace.util.prim [*]])
(import [gace.util.target [*]])
(import [gace.util.render [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass ACE [gym.Env]

  (defn __init__ [self ^int max-steps ^(of dict str float) target 
                       ^bool random-target ^bool noisy-target 
                       ^str data-log-path ^str param-log-path]

    ;; Environment Configurations
    (setv self.max-steps max-steps
          self.num-steps 0
          self.data-log-path data-log-path
          self.data-log (pd.DataFrame)
          self.param-log-path param-log-path)

    ;; If a target was provided, use it during but add some noise during each iteration.
    (setv self.random-target random-target
          self.noisy-target  noisy-target
          self.target        (or target 
                                 (target-specification self.ace-id self.ace-backend
                                      :random self.random-target
                                      :noisy self.noisy-target))
          self.condition (reward-condition self.ace-id 
                                          :tolerance (if (hasattr self "reltol")
                                                         self.reltol 1e-3)))

    (.__init__ (super ACE self)))
  
  (defn reset ^np.array [self]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (unless self.ace
      (setv self.ace (self.ace-constructor)))

    ;; Reset the step counter and increase the reset counter.
    (setv self.num-steps     (int 0))

    ;; Clear the data log. If self.log-data == True the data will be written to
    ;; an HDF5 in the `done` function, otherwise it will be discarded.
    (setv self.data-log (pd.DataFrame))

    ;; Target can be random or close to a known acheivable.
    (setv self.target (if self.random-target
      (target-specification self.ace-id self.ace-backend 
                            :random self.random-target 
                            :noisy self.noisy-target)
      (dfor (, p v) (.items self.target) 
            [p (* v (if self.noisy-target (np.random.normal 1.0 0.01) 1.0))])))

    ;; Starting parameters are either random or close to a known solution.
    (setv parameters (starting-point self.ace self.random-target self.noisy-target))

    ;; Get the current performance for the initial parameters
    (setv performance (ac.evaluate-circuit self.ace :params parameters))

    (observation performance self.target))

    (defn size-circuit [self sizing]
      (let [performance (ac.evaluate-circuit self.ace :params sizing) 
            
            obs (observation performance self.target)
            rew (reward performance self.target self.condition)
            don (or (> self.num-steps self.max-steps) 
                    (all (second (target-distance performance 
                                                  self.target 
                                                  self.condition))))
            inf (info performance self.target) ]

        (when (bool self.data-log-path)
          (setv self.data-log (.append self.data-log (| sizing performance)
                                       :ignore-index True)))

        (when (and (bool self.param-log-path) 
                   (or (np.any (np.isnan obs)) 
                       (np.any (np.isinf obs))))
          (save-state self.ace self.ace-id self.param-log-path))
         
        (when (and (bool self.data-log-path) don)
          (save-data self.data-log self.data-log-path self.ace-id))

      (setv self.num-steps (inc self.num-steps))
      (, obs rew don inf)))

  (defn render [self &optional ^str [mode "human"]]
    (print (ascii-schematic self.ace-id)))

  (defn seed [self rng-seed]
    """
    Sets The RNG Seed for this environment.
    """
    (.seed np.random rng-seed)
    (.manual-seed pt rng-seed)
    rng-seed)

  (defn close [self]
    """
    Closes the spectre session.
    """
    (.stop self.ace)
    (del self.ace)
    (setv self.ace None)))
