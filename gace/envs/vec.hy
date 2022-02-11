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
          self.pool (ac.to-pool (lfor e self.gace-envs e.ace)))

    (setv self.action-space (lfor env self.gace-envs env.action-space))
    (setv self.observation-space (lfor env self.gace-envs env.observation-space))

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
              ;; If ace does not exist, create it.
              :do (unless e.ace (setv e.ace (eval e.ace-constructor)))

              ;; Reset the step counter and increase the reset counter.
              :do (setv e.num-steps (int 0))

              ;; Clear the data log. If self.log-data == True the data will be written to
              ;; an HDF5 in the `done` function, otherwise it will be discarded.
              :do (setv e.data-log (pd.DataFrame))

              ;; Target can be random or close to a known acheivable.
              :do (setv e.target 
                          (if e.random-target
                              (target-specification e.ace-id e.ace-backend 
                                                    :random e.random-target 
                                                    :noisy e.noisy-target)
                              (dfor (, p v) (.items e.target) 
                                 [p (* v (if e.noisy-target 
                                             (np.random.normal 1.0 0.01) 
                                             1.0))])))

              ;; Starting parameters are either random or close to a known solution.
              (starting-point e.ace e.random-target e.noisy-target))))

        performances (ac.evaluate-circuit-pool self.pool 
                                               :pool-params parameters 
                                               :npar self.n-proc)]

    (list (ap-map (observation #* it) (zip (.values performances) targets)))))
      
  (defn size-circuit-pool [self sizings]
    (let [(, prev-perfs targets conds reward-fns inputs) 
          (zip #* (lfor e self.gace-envs (, (ac.current-performance e.ace) 
                                       e.target e.condition e.reward 
                                       e.input-parameters)))
             
          curr-perfs (-> self (. pool) 
                              (ac.evaluate-circuit-pool :pool-params sizings 
                                                        :npar self.n-proc) 
                              (.values))

          obs (lfor (, cp tp) (zip curr-perfs targets)
                    (observation cp tp))
          
          steps (lfor e self.gace-envs e.num-steps)

          rew (lfor (, rf cp pp t c s) 
                    (zip reward-fns curr-perfs prev-perfs targets conds steps)
                    (rf cp pp t c s))

          td  (list (ap-map (-> (target-distance #* it) (second) (all)) 
                            (zip curr-perfs targets conds)))
          ss  (lfor e self.gace-envs (>= (inc e.num-steps) e.max-steps))
          don (list (ap-map (or #* it) (zip td ss))) 

          inf (list (ap-map (info #* it) (zip curr-perfs targets inputs)))]

      (for [(, e s p o d) (zip self.gace-envs (.values sizings) curr-perfs obs don)] 
        (setv e.num-steps (inc e.num-steps))
        (setv e.data-log (.append e.data-log (| s p) :ignore-index True))
        
        (when (and (bool e.param-log-path) 
                   (or (np.any (np.isnan o)) 
                       (np.any (np.isinf o))))
        (save-state e.ace e.ace-id e.param-log-path))
       
        (when (and (bool e.data-log-path) d)
          (save-data e.data-log e.data-log-path e.ace-id)))

      (, obs rew don inf)))

  (defn seed [self rng-seed &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.seed rng-seed)))

  (defn close [self &optional ^(of list int) [env-ids []]]
    (lfor e (if env-ids (lfor i env-ids (get self.gace-envs i)) 
                        self.gace-envs) 
         (e.close))))
