(import os)
(import sys)
(import errno)
(import datetime)
(import [functools [partial]])
(import [fractions [Fraction]])

(import [numpy :as np])
(import [pyarrow :as pa])
(import [pyarrow [feather :as ft]])

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

(defn vector-make [envs &optional ^int [n-proc DEFAULT_N_PROC]]
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
  (vector-make (list (take num-envs (repeatedly #%(gym.make env-id #** kwargs)))) n-proc))

(defclass VecACE []
  (defn __init__ [self envs ^int n-proc]
    (setv self.n-proc n-proc
          self.gace-envs envs
          self.num-envs  (len envs)
          self.pool (ac.to-pool (lfor e self.gace-envs e.ace)))

    (setv self.action-space (lfor env self.gace-envs env.action-space))
    (setv self.observation-space (lfor env self.gace-envs env.observation-space))

    ;; Environment Logging
    (setv time-stamp (-> datetime (. datetime) (.now) (.strftime "%Y%m%d-%H%M%S"))
          self.base-log-path f"/tmp/{(.getlogin os)}/gace/{time-stamp}-pool")
    ;(for [(, i env) (enumerate self.gace-envs)]
    ;  (setv env.data-log-path f"{self.base-log-path}/env_{i}"))

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

  (defn reset ^np.array [self &optional ^(of list int) [env-ids []]]
    """
    If not running, this creates a new spectre session. The `moves` counter is
    reset, while the reset counter is increased. If `same-target` is false, a
    random target will be generated otherwise, the given one will be used.

    Finally, a simulation is run and the observed perforamnce returned.
    """

    (let [envs (if env-ids (lfor i env-ids (get self.gace-envs i)) self.gace-envs)
          targets (lfor e envs e.target)
          parameters (dict (enumerate (lfor e envs
              ;; Reset the step counter and increase the reset counter.
              :do (setv e.env.num-steps (int 0))
              :do (setv e.env.reset-count (inc e.reset-count))

              ;; If ace does not exist, create it.
              ;:do (unless e.env.ace (setv e.env.ace (eval e.env.ace-constructor)))
              :do (when (or (not e.env.ace) 
                            (= 0 (% e.env.reset-count e.env.restart-intervall)))
                    (e.env.ace.stop)
                    (del e.env.ace)
                    (setv e.env.ace (eval e.env.ace-constructor)))

              ;; Target can be random or close to a known acheivable.
              :do (setv e.env.target (target-specification e.env.ace-id e.env.design-constraints
                                                    :random e.env.random-target 
                                                    :noisy e.env.noisy-target))

              ;; Starting parameters are either random or close to a known solution.
              (starting-point e.env.ace e.env.random-target e.env.noisy-target))))

        performances (ac.evaluate-circuit-pool self.pool 
                                               :pool-params parameters 
                                               :npar self.n-proc)]

    ;;Target Logging 
    (for [(, i e) (enumerate self.gace-envs)]
      (when e.env.logging-enabled 
        (e.env.log-target :log-path (.format "{}/env_{}" self.base-log-path i))))

    (setv self.info 
        (lfor (, p (, t i)) (zip (.values performances)
                                 (lfor e envs (, e.env.target e.env.input-parameters)))
              (info p t i) ))

    (list (ap-map (observation #* it) (zip (.values performances) 
                                           targets (repeat 0) 
                                           (lfor e envs e.env.max-steps))))))

  (defn size-circuit-pool [self sizings]
    (let [(, prev-perfs targets conds reward-fns inputs steps max-steps
            last-actions) 
                (zip #* (lfor e self.gace-envs (, (ac.current-performance e.ace) 
                                               e.env.target e.env.condition e.env.reward 
                                               e.env.input-parameters
                                               e.env.num-steps e.env.max-steps
                                               e.env.last-action)))
             
          curr-perfs (-> self (. pool) 
                              (ac.evaluate-circuit-pool :pool-params sizings 
                                                        :npar self.n-proc) 
                              (.values))
          
          curr-sizings (-> self (. pool) (ac.current-sizing-pool) (.values))
          set-sizings  (.values sizings)

          obs (lfor (, cp tp ns ms) (zip curr-perfs targets steps max-steps)
                    (observation cp tp ns ms))

          rew (lfor (, rf cp pp t c cs ss a s m) 
                    (zip reward-fns curr-perfs prev-perfs targets conds 
                         curr-sizings set-sizings last-actions steps max-steps)
                    (rf cp pp t c cs ss a s m))

          td  (list (ap-map (-> (target-distance #* it) (second) (all)) 
                            (zip curr-perfs targets conds)))

          ss  (lfor e self.gace-envs (>= (inc e.env.num-steps) e.env.max-steps))
          don (list (ap-map (or #* it) (zip td ss))) 

          inf (list (ap-map (info #* it) (zip curr-perfs targets inputs)))]

      ;; Increment step counter
      (for [e self.gace-envs] (setv e.env.num-steps (inc e.env.num-steps)))

      ;; Data Logging
      (for [(, i e s p) (zip (-> self.num-envs (range) (list)) 
                             self.gace-envs curr-sizings curr-perfs)]
        (when e.env.logging-enabled 
          (e.env.log-data s p (.format "{}/env_{}" self.base-log-path i))))

      (, obs rew don inf)))

  (defn seed [self rng-seed &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.seed rng-seed)))

  (defn close [self &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.close))))
