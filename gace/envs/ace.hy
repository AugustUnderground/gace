(import os)
(import sys)
(import errno)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import gym)

(import [gace.util.func [*]])
(import [gace.util.prim [*]])
(import [gace.util.target [*]])
(import [gace.util.render [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import  [typing [List Set Dict Tuple Optional Union Callable]])
(import  [hy.contrib.sequences [Sequence end-sequence]])
(import  [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass ACE [gym.Env]
  """
  Base class interfacing both ACE and gym.
  Arguments:
    ace-id: str                         -> ACE Identifier ∈ [op#, st1, nand4]
    ace-backend: str                    -> ACE Backend ∈ [xh035, sky130, gpdk180]
  Optional:
    ckt-path: str (None)                -> Path to ACE Testbench
    pdk-path: str (None)                -> Path to PDK
    obs-lo: Union[float, np.array] -Inf -> Lower bound of obs
    obs-hi: Union[float, np.array] +Inf -> Upper bound of obs
    max-steps: int (666)                -> Maximum number of steps before reset
    target: Dict[str, float] ({})       -> Specific Target to reach
    random-target: bool (False)         -> Randomize Target each episode
    noisy-target: bool  (True)          -> Add some noise after each reset
    train-mode: bool (True)             -> Whether this is training or eval
    custom-reward: function (None)      -> A custom reward function
    reltol: float (1e-3)                -> Relative tolarnce for equaltiy
    data-log-path: str ("")             -> Write a dataframe to HDF5 at this location
    param-log-path: str ('.')           -> Write a json to this location
  """
  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self ^str ace-id ^str ace-backend ^int ace-variant &optional 
                       ^str [ckt-path None] ^str [pdk-path None]
                       ^(of Union float np.array) [obs-lo (- Inf)]
                       ^(of Union float np.array) [obs-hi Inf]
                       ^int [max-steps 666] ^(of dict str float) [design-constr {}]
                       ^(of dict str float) [target {}] ^float [reltol 1e-3]
                       ^bool [random-target False] ^bool [noisy-target True]
                       ^bool [train-mode True] 
                       ^(of Callable)   [custom-reward None]
                       ^(of gym.spaces) [custom-action None]
                       ^(of np.array)   [custom-action-lo None]
                       ^(of np.array)   [custom-action-hi None]
                       ^str [nmos-path None] ^str [pmos-path None]
                       ^str [data-log-path ""] ^str [param-log-path "."]]

    ;; ACE Configuration
    (setv self.ace-id          ace-id
          self.ace-backend     ace-backend
          self.ace-variant     ace-variant
          self.ace-constructor (ace-constructor self.ace-id self.ace-backend 
                                                :ckt ckt-path :pdk [pdk-path])
          self.ace             (self.ace-constructor))

    ;; Obtain design constraints from ACE backend and override if given
    (setv dc (design-constraints self.ace)
          self.design-constraints (dfor k (.keys dc) 
                                          [k (.get design-constr k 
                                                   (get dc k))]))

    ;; Generate action space for the given ACE Environment
    (setv (, self.action-space  
          self.action-scale-min
          self.action-scale-max) (if custom-action 
                                    (, custom-action 
                                       custom-action-lo 
                                       custom-action-hi) 
                                    (action-space self.ace
                                                  self.design-constraints 
                                                  self.ace-id 
                                                  self.ace-variant)))

    ;; Default Step functions
    (setv self.step-funcs (| {1 (fn [^np.array action &optional [blocklist []]]
                                  (-> (step-v1 self.input-parameters 
                                               self.action-scale-min 
                                               self.action-scale-max 
                                               action)
                                     (self.size-circuit :blocklist blocklist)))
                              3 (fn [^np.array action &optional [blocklist []]]
                                  (-> (step-v3 self.input-parameters 
                                               self.design-constraints
                                               self.ace action)
                                      (self.size-circuit :blocklist blocklist)))
                              #_/ }
                              (if (hasattr self "step_v0") 
                                  {0 self.step-v0} {})
                              (if (hasattr self "step_v2") 
                                  {2 self.step-v2} {})))

    ;; Override step function
    (setv self.step (get self.step-funcs self.ace-variant))

    ;; Set training mode by default
    (setv self.train-mode train-mode)

    ;; Environment Configurations
    (setv self.max-steps max-steps
          self.num-steps (int 0)
          self.data-log-path data-log-path
          self.data-log (pd.DataFrame)
          self.param-log-path param-log-path)

    ;; If a target was provided, use it during but add some noise during each iteration.
    (setv self.random-target random-target
          self.noisy-target  noisy-target
          self.target        (or target 
                                 (target-specification self.ace-id 
                                                       self.design-constraints
                                                       :random self.random-target
                                                       :noisy self.noisy-target))
          self.reltol        reltol
          self.reward        (or custom-reward reward)
          self.condition     (reward-condition self.ace-id :tolerance self.reltol))

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv obs-shape (observation-shape self.ace self.ace-id 
                                       (-> self.target (.keys) (list)))
          self.observation-space (Box :low obs-lo :high obs-hi 
                                      :shape obs-shape
                                      :dtype np.float32))

    ;; Primitive Device setup
    (when (in self.ace-variant [0 2])
      (setv self.nmos (load-primitive "nmos" self.ace-backend :dev-path nmos-path)
            self.pmos (load-primitive "pmos" self.ace-backend :dev-path pmos-path)))

    ;; Specify Input Parameter Names
    (setv self.input-parameters (input-parameters self.ace self.ace-id 
                                                  self.ace-variant))

    ;; Call gym.Env constructor
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

  (defn size-circuit [self sizing &optional [blocklist []]]
    (let [performance (ac.evaluate-circuit self.ace :params sizing
                                                    :blocklist blocklist) 
          
          obs (observation performance self.target)
          rew (self.reward performance self.target self.condition self.reltol)
          don (or (>= (inc self.num-steps) self.max-steps) 
                  (all (second (target-distance performance 
                                                self.target 
                                                self.condition))))
          inf (info performance self.target self.input-parameters) ]

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
