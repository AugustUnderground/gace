(import os)
(import sys)
(import errno)
(import datetime)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [numpy :as np])

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

(setv DEFAULT_N_PROC (-> 0 (os.sched-getaffinity) (len) (// 2)))

(defn vector-make [^list envs &optional ^int [n-proc DEFAULT_N_PROC]]
  """
  Takes a list of gace environments and returns a 'vectorized' version thereof.
  """
  (VecACE envs n-proc))

(defn vector-make-same [^str env-id ^int num-envs 
        &optional ^int [n-proc DEFAULT_N_PROC] &kwargs kwargs]
  """
  Takes a gace environment id and a number and returns a vectorized
  environemnt, with n times the given id. 
    Short hand for: `vector_make([gym.make(env_id) for _ range(num_envs)])`
  """
  (vector-make (lfor _ (range num-envs) 
                     (-> env-id (gym.make #** kwargs) (. unwrapped))) 
               :n-proc n-proc))

(defclass VecACE []
  (defn __init__ [self ^list envs ^int n-proc]
    (setv self.n-proc    n-proc
          self.gace-envs envs
          self.num-envs  (len envs)
          ;self.ace-envs  (dfor (, i e) (enumerate self.gace-envs) [i e.ace])
          #_/ )

    (setv self.action-space      (lfor e self.gace-envs e.action-space))
    (setv self.observation-space (lfor e self.gace-envs e.observation-space))

    ;; Environment Logging
    (setv time-stamp (-> datetime (. datetime) (.now) (.strftime "%Y%m%d-%H%M%S"))
          self.base-log-path f"/tmp/{(.getlogin os)}/gace/{time-stamp}-pool")
    (for [(, i e) (enumerate self.gace-envs)]
      (when e.logging-enabled
        (os.system f"rm -rf {e.data-log-path}")
        (setv e.data-log-path (.format "{}/env_{}" self.base-log-path i)
              e.data-logger (initialize-data-logger e.ace e.ace-variant e.target 
                                                    e.input-parameters 
                                                    e.data-log-path))))

    (setv self.step 
          (fn [^(of list np.array) actions]
            (let [sizings (->> actions (zip self.gace-envs)
                                       (ap-map (-> it (first) (.step-fn (second it))))
                                       (enumerate) (dict))]
              (self.size-circuit-pool sizings)))))

  (defn __len__ [self] 
    """
    Returns the number of Environments in the Vector.
    """
    (len self.gace-envs))

  (defn __iter__ [self]
  """
  Make VecEnv Iterable.
  """
    (-> self (. gace-envs) (iter)))

  (defn random-step [self]
  """
  Vectorized version of the convenience function in case you say to yourself:
  'But I just wanna take a random step'.
  """
    (self.step (lfor as self.action-space (-> as (.sample)))))

  (defn reset ^np.array [self &optional ^(of list int) [env-ids []]
                                        ^(of list bool) [done-mask None]]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.

    Finally, a simulation is run and the observed perforamnce returned.
    """
    (let [envs (cond [env-ids (dfor i env-ids [i (get self.gace-envs i)])]
                     [(and done-mask (= (len done-mask) self.num-envs))
                      (dfor (, i d) (enumerate done-mask) :if d
                            [i (get self.gace-envs i)])]
                     [True (dict (enumerate self.gace-envs))])

          parameters (dfor (, i e) (.items envs)
              ;; Reset the step counter and increase the reset counter.
              ;:do (setv e.num-steps (int 0))
              :do (setv e.reset-count (inc e.reset-count))

              ;; If ace does not exist, create it.
              :do (when (or (not e.ace) (= 0 (% e.reset-count e.restart-intervall)))
                    (.clear e.ace)
                    ;(del e.ace)
                    (setv e.ace (eval e.ace-constructor)))
              ;:do (unless e.ace (setv e.ace (eval e.ace-constructor)))

              ;; Target can be random or close to a known acheivable.
              :do (setv e.target (target-specification e.ace-id 
                                                       e.design-constraints
                                                       e.target-filter
                                                       :random (or e.random-target 
                                                                  (> e.reset-count 
                                                                     1000))
                                                       :noisy e.noisy-target))

              ;; Log new target
              :do (when e.logging-enabled
                    ;(setv e.data-log (initialize-data-log e.ace e.target e.reset-count))
                    (e.log-target e.target))

              ;; Starting parameters are either random or close to a known solution.
              [i (starting-point e.ace e.ace-variant e.reset-count e.num-steps 
                                 e.max-steps e.design-constraints 
                                 e.random-target e.noisy-target)])

          ace-envs (dfor (, i e) (enumerate self.gace-envs) [i e.ace])

          ;; Only simulate sub-pool of reset envs
          performances (if parameters
                           (ac.evaluate-circuit-pool ace-envs
                                                     :pool-params parameters
                                                     :pool-ids env-ids
                                                     :npar self.n-proc)
                           (ac.current-performance-pool ace-envs)) ]

    ;; Reset Step counters
    (for [(, i e) (.items envs)] (setv e.num-steps (int 0)) )

    ;; Targets of pooled envs
    (setv self.targets (lfor e self.gace-envs e.target))

    (setv self.info 
        (lfor (, p (, t i)) (zip (.values performances)
                                 (lfor e self.gace-envs (, e.target e.input-parameters)))
              (info p t i)))

    (list (ap-map (observation #* it) 
                  (zip (.values performances) 
                       #* (zip #* (lfor e self.gace-envs 
                                        (, e.target e.num-steps e.max-steps))))))))

  (defn size-circuit-pool [self sizings]
    (let [(, targets conds reward-fns inputs steps max-steps last-actions) 
                (zip #* (lfor e self.gace-envs (, e.target e.condition 
                                                  e.reward 
                                                  e.input-parameters
                                                  e.num-steps e.max-steps
                                                  e.last-action)))

          ace-envs (dfor (, i e) (enumerate self.gace-envs) [i e.ace])

          prev-perfs (-> ace-envs (ac.current-performance-pool) (.values))
             
          curr-perfs (-> ace-envs (ac.evaluate-circuit-pool :pool-params sizings 
                                                            :npar self.n-proc) 
                                  (.values))
          
          curr-sizings (-> ace-envs (ac.current-sizing-pool) (.values))

          set-sizings  (.values sizings)

          obs (lfor (, cp tp ns ms) (zip curr-perfs targets (map inc steps) max-steps)
                    (observation cp tp ns ms))

          rew (lfor (, rf cp pp t c cs ss a s m) 
                    (zip reward-fns curr-perfs prev-perfs targets conds 
                         curr-sizings set-sizings last-actions steps max-steps)
                    (rf cp pp t c cs ss a s m))

          td  (list (ap-map (-> (target-distance #* it) (second) (np.all) (bool)) 
                            (zip curr-perfs targets conds)))

          ss  (lfor e self.gace-envs (>= e.num-steps e.max-steps))

          don (list (ap-map (or #* it) (zip td ss))) 

          inf (list (ap-map (info #* it) (zip curr-perfs targets inputs)))]

      ;; Increment step counter
      (for [e self.gace-envs] (setv e.num-steps (inc e.num-steps)))

      ;; Data Logging
      (for [(, i e s p r) (zip (-> self.num-envs (range) (list)) 
                             self.gace-envs curr-sizings curr-perfs rew)]
        (when e.logging-enabled 
          (e.log-data s p r)))
          ;(e.log-data s p r (.format "{}/env_{}" self.base-log-path i))))

      (, obs rew don inf)))

  (defn seed [self rng-seed &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.seed rng-seed)))

  (defn close [self &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.close))))
