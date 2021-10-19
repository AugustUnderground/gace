(import os)
(import sys)
(import errno)
(import [functools [partial]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import gym)
(import [gym.spaces [Dict Box Discrete MultiDiscrete Tuple]])

(import [pettingzoo [AECEnv ParallelEnv]])
(import [pettingzoo.utils [agent-selector wrappers from-parallel]])

(import ray)
(import [ray.rllib.env.multi-agent-env [MultiAgentEnv]])

(import [.amp_env [AmplifierXH035Env]])
(import [.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defn sym-env [&kwargs kwargs]
  (-> (sym-raw #** kwargs)
      (wrappers.CaptureStdoutWrapper)
      ;(wrappers.AssertOutOfBoundsWrapper)
      (wrappers.OrderEnforcingWrapper)))

(defn sym-raw [&kwargs kwargs]
  (-> (SymAmpXH035ZooMA #** kwargs)
      (from-parallel)))

(defclass SymAmpXH035ZooMA [ParallelEnv SymAmpXH035Env]
  (setv metadata {"render.modes" ["human"] "name" "sym-amp-xh035-v1"})

  (defn __init__ [self &kwargs kwargs]
    (SymAmpXH035Env.__init__ self #** kwargs)

    (setv self.possible-agents ["pcm_2" "ndp_1" "ncm_1" "ncm_3"]
          self.agent-name-mapping (dict (zip self.possible-agents
                                             (-> self.possible-agents 
                                                 (len) (range) (list)))))
    
    (setv self.action-spaces 
            { "pcm_2" (Box :low -1.0 :high 1.0 :shape (, 3)
                           :dtype np.float32)
              "ndp_1" (Box :low -1.0 :high 1.0 :shape (, 3)
                           :dtype np.float32)
              "ncm_1" (Box :low -1.0 :high 1.0 :shape (, 3)
                           :dtype np.float32)
              "ncm_3" (Box :low -1.0 :high 1.0 :shape (, 3)
                           :dtype np.float32) })

    (setv self.observation-spaces 
            (dfor agent self.possible-agents
              [agent (Box :low (np.nan-to-num (- np.inf)) 
                          :high (np.nan-to-num np.inf)
                          :shape (, 201) :dtype np.float32)])))

  (defn observe [self agent]
    (get self.observations agent))

  (defn reset [self]
    ;; Reset Agent cycle
    (setv self.agents (get self.possible-agents (slice None None)))

    ;; Get observation from parent class
    (setv obs (SymAmpXH035Env.reset self))
    (dfor agent self.agents [agent obs]))

  (defn observation-space [self agent]
    (get self.observation-spaces agent))
  
  (defn action-space [self agent]
    (get self.action-spaces agent))

  (defn _forall-agents [self same-value]
    (dfor agent self.agents
          [agent same-value]))

  (defn step [self actions]
    (let [(, gmid-cm1 fug-cm1 mcm1) (get actions "ncm_1")
          (, gmid-cm2 fug-cm2 mcm2) (get actions "pcm_2")
          (, gmid-dp1 fug-dp1 _)    (get actions "ndp_1")
          (, gmid-cm3 fug-cm3 _)    (get actions "ncm_3")
          action (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
                            fug-cm1  fug-cm2  fug-cm3  fug-dp1 
                            mcm1 mcm2])
          (, o r d i) (SymAmpXH035Env.step self action) 
          observations (self._forall-agents o)
          rewards (self._forall-agents r)
          dones (self._forall-agents d)
          infos (self._forall-agents i) ]
      (, observations rewards dones infos))))

(defclass SymAmpXH035RayMA [MultiAgentEnv SymAmpXH035Env]
  (defn __init__ [self config]
    (SymAmpXH035Env.__init__ self #** config)
    (setv self.building-blocks ["pcm_2" "ndp_1" "ncm_1" "ncm_3"]))

  (defn _forall-bb [self same-value]
    (dfor bb self.building-blocks
          [bb same-value]))

  #@(staticmethod
  (defn policy-config ^tuple []
    (let [obs-space (Box :low (np.nan-to-num (- np.inf)) 
                         :high (np.nan-to-num np.inf)
                         :shape (, 203) :dtype np.float32)
        act-space-dp (Box :low -1.0 :high 1.0 :shape (, 2) :dtype np.float32)
        act-space-cm (Box :low -1.0 :high 1.0 :shape (, 3) :dtype np.float32)

        policies { "ncm" (, None obs-space act-space-cm {})
                   "pcm" (, None obs-space act-space-cm {})
                   "ndp" (, None obs-space act-space-cm {})
                   "pdp" (, None obs-space act-space-cm {}) }
        policy-mapping (fn [a e w &kwargs kw] (get a (slice 0 3))) ]
      (, policies policy-mapping))))

  (defn reset [self]
    (let [obs (SymAmpXH035Env.reset self)]
      (self._forall-bb obs)))

  (defn step ^dict [self ^dict actions]
    (let [(, gmid-cm1 fug-cm1 mcm1) (get actions "ncm_1")
          (, gmid-cm2 fug-cm2 mcm2) (get actions "pcm_2")
          (, gmid-dp1 fug-dp1)      (get actions "ndp_1")
          (, gmid-cm3 fug-cm3 _)    (get actions "ncm_3")
          action                    (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
                                               fug-cm1  fug-cm2  fug-cm3  fug-dp1 
                                               mcm1 mcm2])
          (, o r d i)               (SymAmpXH035Env.step self action) 
          observations              (self._forall-bb o)
          rewards                   (self._forall-bb r)
          dones                     (self._forall-bb d)
          _                         (setv (get dones "__all__") 
                                          (-> dones (.values) (all)))
          infos                     (self._forall-bb i)]
      (, observations rewards dones infos))))
