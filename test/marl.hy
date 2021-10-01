(import os)
(import [threading [Thread]])
(import [functools [partial]])
(import [operator [itemgetter]])
(import [datetime [datetime :as dt]])
(import [numpy :as np])
(import [gym-ad.envs [SymAmpXH035ZooMA SymAmpXH035RayMA]])
(import [supersuit :as ss])
(import [stable-baselines3 [A2C PPO]])
(import [stable-baselines3.common.vec-env [DummyVecEnv VecNormalize SubprocVecEnv]])
(import ray)
(import [ray.tune [register-env]])
(import [ray.rllib.env.policy-server-input [PolicyServerInput]])
(import [ray.rllib.env.policy-client [PolicyClient]])
(import [ray.rllib.examples.env.random-env [RandomMultiAgentEnv]])
(import [ray.rllib.agents.a3c :as a3c])
(import [ray.rllib.agents.ppo :as ppo])
(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; RAY
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      algorithm-name f"a3c"
      model-path f"./models/baselines/ray-{algorithm-name}-sym-amp-xh035-{time-stamp}.mod"
      data-path  f"../data/symamp/xh035"
      nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos"
      host      "localhost"
      port      6666)

(.init ray)

(register-env "ad-server" (fn [c] (RandomMultiAgentEnv c)))

(setv (, policies policy-mapping) (.policy-config SymAmpXH035RayMA))

(setv env-cfg {"nmos_path"       nmos-path
               "pmos_path"       pmos-path
               "data_log_prefix" data-path
               "close_target"    True }
      config  {"input" (fn [ioctx] (PolicyServerInput ioctx host port))
               "num_workers" 0
               "input_evaluation" []
               "train_batch_size" 256
               "rollout_fragment_length" 20
               "multiagent" { "policies" policies
                              "policy_mapping_fn" policy-mapping }
               "framework" "torch"
               "env_config" env-cfg})

(setv trainer (ppo.PPOTrainer :env "ad-server" :config config))

(defn mad-server []
  (loop [[brk False] [itr 0]]
    (print f"{itr :04}. Iteration")
    (print (.train trainer))
    (with [f (open model-path "w")]
      (f.write (.save trainer)))
    (recur False (inc itr))))

(setv client (PolicyClient f"http://{host}:{port}" 
                           :inference-mode "local" 
                           :update-interval 10.0))

(setv env (SymAmpXH035RayMA env-cfg)
      init-obs (.reset env)
      reward-tolerance 0.95
      init-eid (client.start-episode :training-enabled True))

(defn mad-client []
  (loop [[brk False] [itr 0] [obs init-obs] [eid init-eid]]
    (let [actions (client.get-action eid obs)
          (, o r d i) (env.step actions)
          total-reward (-> r (.values) (sum))]
      (client.log-returns eid r i :multiagent-done-dict d)
      (if (.get d "__all__" False) 
        (do
          (print (.format "Episode done: Reward = {}" total-reward))
          (when (>= total-reward reward-tolerance)
            (quit 0))
          (client.end-episode eid obs)
          (recur False (inc itr) (.reset env) (client.start-episode :training-enabled True)))
        (recur False (inc itr) o eid)))))

(.start (setx th-server (Thread :target mad-server)))

(setv actions (client.get-action init-eid init-obs))
(setv (, o r d i) (env.step actions))
(setv total-reward (-> r (.values) (sum)))


;(.start (setx th-client (Thread :target mad-client)))
(mad-client)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; ZOO
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      algorithm-name f"ppo"
      model-path f"./models/baselines/marl-{algorithm-name}-sym-amp-xh035-{time-stamp}.mod"
      data-path  f"../data/symamp/xh035"
      nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos")

(setv env (SymAmpXH035ZooMA :nmos-path       nmos-path
                            :pmos-path       pmos-path
                            :data-log-prefix data-path
                            :close-target    True))

(setv venv (ss.pettingzoo-env-to-vec-env-v0 env))
;(setv nenv (VecNormalize venv :training True :norm-obs True :norm-reward True))

(setv model (A2C "MlpPolicy" venv :verbose 1))
(model.learn :total-timesteps 10000 :log-interval 1)
(model.save model-path)


