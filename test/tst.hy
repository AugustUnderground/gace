(import os)
(import [datetime [datetime :as dt]])
(import gym)
(import [torch :as pt])
(import [torch [nn]])
(import [numpy :as np])
(import [torch.utils.tensorboard [SummaryWriter]])
(import [torch.distributions [Independent Normal]])
(import [tianshou :as ts])
(import [tianshou.env [SubprocVectorEnv DummyVectorEnv]])
(import [tianshou.policy [A2CPolicy PPOPolicy TD3Policy]])
(import [tianshou.data [Collector ReplayBuffer VectorReplayBuffer]])
(import [tianshou.trainer [onpolicy-trainer]])
(import [tianshou.utils.net.common [Net]])
(import [tianshou.utils.net.continuous [ActorProb Critic]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv HOME       (os.path.expanduser "~")
      time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
      model-path f"./models/tianshou/sym-amp-xh035-{time-stamp}.mod"
      data-path  f"../data/symamp/xh035.h5")

;; Environment
(setv nmos-path f"../models/xh035-nmos"
      pmos-path f"../models/xh035-pmos"
      pdk-path  f"{HOME}/gonzo/Opt/pdk/x-fab/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      ckt-path  f"../library/"
      num-train 1
      num-valid 1
      task-name "gym_ad:sym-amp-xh035-v0")

(setv env (gym.make task-name
                    :nmos-path    nmos-path
                    :pmos-path    pmos-path
                    :pdk-path     pdk-path
                    :ckt-path     ckt-path
                    :close-target True))

;(setv train-envs (SubprocVectorEnv (list (repeat #%(identity env) num-train)) 
;                                   :norm-obs True)
;      valid-envs (SubprocVectorEnv (list (repeat #%(identity env) num-valid)) 
;                                   :norm-obs True
;                                   :obs-rms train-envs.obs-rms
;                                   :update-obs-rms False))

(setv train-envs (DummyVectorEnv (list (repeat #%(identity env) num-train))
                                 :norm-obs True)
      valid-envs (DummyVectorEnv (list (repeat #%(identity env) num-valid))
                                 :norm-obs True
                                 :obs-rms train-envs.obs-rms
                                 :update-obs-rms False))

;; Seed
(setv rng-seed 666)
(.seed np.random rng-seed)
(.manual-seed pt rng-seed)
(.seed train-envs rng-seed)
(.seed valid-envs rng-seed)

;; Model
(setv λ 1e-3 ε 1e-5 α 0.99 γ 0.99
      λ-gae 0.95
      λ-scheduler None
      vf-coef 0.5
      ent-coef 0.01
      max-grad-norm None
      reward-norm False
      max-batch-size 1
      action-scaling True
      bound-method "tanh"
      deterministic-eval False
      buffer-size 1
      max-action (first env.action-space.high))

(setv state-shape env.observation-space.shape
      action-shape env.action-space.shape
      hidden-shape [256 128 64]
      device (if (.is-available pt.cuda) "cuda" "cpu"))

;; Actor
(setv actor-net (Net state-shape :hidden-sizes hidden-shape :activation nn.Tanh :device device)
      actor (-> actor-net 
               (ActorProb action-shape :max-action max-action :unbounded True :device device) 
               (.to device)))

(nn.init.constant_ actor.sigma-param -0.5)

;; Critic
(setv critic-net (Net state-shape :hidden-sizes hidden-shape :activation nn.Tanh :device device)
      critic (-> critic-net (Critic :device device) (.to device)))

;; Orthogonal Initialization
(for [m (reduce + (map #%(-> %1 (.modules) (list)) [actor critic]) [])]
  (when (isinstance m nn.Linear)
    (nn.init.orthogonal_ m.weight :gain (np.sqrt 2))
    (nn.init.zeros_ m.bias)))

;; Last Policy Layer Scaling: https://arxiv.org/abs/2006.05990 (Fig.24)
(for [m (.modules actor.mu)]
  (when (isinstance m nn.Linear)
    (nn.init.zeros_ m.bias)
    (m.weight.data.copy_ (* 0.01 m.weight.data))))

;; Optimizer
(setv optim (pt.optim.RMSprop (reduce + (map #%(-> %1 (.parameters) (list)) [actor critic]) [])
                              :lr λ :eps ε :alpha α))

;; Distribution
(defn dist [&rest logits]
  (Independent (Normal #* logits) 1))

;; Policy
(setv policy (A2CPolicy actor critic optim dist 
                        :discount-factor γ 
                        :vf-coef vf-coef
                        :ent-coef ent-coef
                        :max-grad-norm max-grad-norm
                        :gae-lambda λ-gae
                        :reward-normalization reward-norm
                        :max-batchsize max-batch-size
                        :action-scaling action-scaling
                        :action-bound-method bound-method
                        :action-space env.action-space
                        :lr-scheduler λ-scheduler
                        :deterministic-eval deterministic-eval))

;; Replay Buffer
(setv buffer (VectorReplayBuffer buffer-size (len train-envs))
      train-collector (Collector policy train-envs buffer :exploration-noise True)
      valid-collector (Collector policy valid-envs))

;; Logging
(setv log-file f"./logs/a2c-{time-stamp}"
      writer (SummaryWriter "./logs/a2c")
      logger (ts.utils.TensorboardLogger writer 
                                         :update-interval 1 
                                         :train-interval 1))

;; Save callback
(defn save-callback [policy]
  (-> policy (.state-dict) (pt.save model-path)))

;; Training
(setv num-epochs 10
      steps-per-epoch 3
      steps-per-collect 8
      repeat-per-collect 1
      batch-size 5)

(setv train-result (onpolicy-trainer policy train-collector valid-collector
                                     num-epochs steps-per-epoch 
                                     repeat-per-collect num-valid batch-size
                                     :step-per-collect steps-per-collect 
                                     :save-fn save-callback 
                                     :logger logger
                                     :test-in-train False
                                     :verbose True))

;; Evaluation
(.eval policy)
(.seed valid-envs rng-seed)
(.reset valid-collector)

(setv valid-result (.collect valid-collector :n-episode valid-num :render False))

(print f"Final reward : {(-> valid-result (get "rews") (.mean))}, length: {(-> valid-result (get "lens") (.mean))}")
