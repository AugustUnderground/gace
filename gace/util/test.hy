(import gym)
(import warnings)
(import [numpy :as np])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defn check-env [env]
  """
  Partially stolen from 
  https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_checker.py

  Check that the given `env` is a compatible gym.Env and follows the `gace`
  API.
  """
  (assert (isinstance env gym.Env) "The env must inherit from gym.Env class.")

  (assert (hasattr env "observation_space") "The env must specify an observation_space.")
  (assert (hasattr env "action_space") "The env must specify an action_space.")
  (assert (isinstance env.observation-space gym.spaces.Space) 
          "The observation space must inherit from gym.spaces.")
  (assert (isinstance env.action-space gym.spaces.Space)
          "The action space must inherit from gym.spaces.")
  
  (when (isinstance env.action-space gym.spaces.Dict)
    (for [s (env.action-space.values)]
      (assert (isinstance s gym.spaces.Space) 
          "The observation space must inherit from gym.spaces.")))

  (when (not-in (len env.observation-space.shape) [1 3])
    (warnings.warn "The observation space is not ∈ [1,3], it should be flattened."))

  (when (np.any (!= (np.abs env.action-space.low) (np.abs env.action-space.high)))
    (warnings.warn "The action space is not symmetric."))
  
  (when (or (np.any (> (np.abs env.action-space.low) 1) )
            (np.any (> (np.abs env.action-space.high) 1)))
    (warnings.warn "The action space is not ∈ [-1;1]."))

  (when  (not (env.action-space.is-bounded))
    (warnings.warn "The action space is not bounded."))

  (setv obs (env.reset))

  (assert (isinstance obs np.ndarray) 
          "The observation returned by 'reset' must be a numpy array.")

  (assert (env.observation-space.contains obs) 
    f"'reset' returned observations:\n{obs}
      not matching the given space: {obs.shape} ≠ {env.observation-space.shape}")

  (setv act (.sample env.action-space)
        res (.step env act))
  
  (assert (= (len res) 4) 
    "The 'step' function must return four values: observation, reward, done, info.")

  (setv (, o r d i) res)

  (assert (isinstance o np.ndarray) 
          "The observation returned by 'reset' must be a numpy array.")
  (assert (env.observation-space.contains o) 
    f"'reset' returned observations:\n{obs}\nnot matching the given space.")

  (assert (isinstance r (, float int)) f"The reward returned by 'step' must be float not {r}.")
  (assert (isinstance d bool) f"The done signal returned by 'step' must be bool not {d}.")
  (assert (isinstance i dict) f"The info returned by 'step' must be a dict not {i}.")

  (setv modes (env.metadata.get "render.modes"))
  (if modes
    (for [m modes] (env.render :mode m))
    (warnings.warn "The environment declares no render modes."))

  (for [_ (range 10)]
    (let [act (.sample env.action-space)
          (, obs rew don inf) (.step env act)]
      (when (-> obs (np.isnan) (np.any))
        (warnings.warn (.format "Received NaN in observations for:\n{}\n{}" act obs)))
      (when (-> obs (np.isinf) (np.any))
        (warnings.warn (.format "Received Inf in observations for:\n{}\n{}" act obs))))))
