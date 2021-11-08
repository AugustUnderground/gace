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

(import [.trg [TriggerEnv]])
(import [.util.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass ST1Env [TriggerEnv]
  """
  Derived amplifier class, implementing the Miller Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-start True]
                                 ^bool [random-target False]
                                 ^float [tolerance 1e-3]
                                 ^str [data-log-prefix ""]]
    """
    Constructs a Miller Amplifier Environment with XH035 device models and
    the corresponding netlist.
    Arguments:
      pdk-path:   This will be passed to the ACE backend.
      ckt-path:   This will be passed to the ACE backend.
      max-moves:  Maximum amount of steps the agent is allowed to take per
                  episode, before it counts as failed. Default = 200.
      random-start: Generate new random starting point for each episode.
      tolerance:  Tolerance for reaching target.
    """

    ;; ACE Environment ID
    (setv self.ace-env "st1")

    ;; Check given paths
    (unless (or pdk-path (not (os.path.exists pdk-path)))
      (raise (FileNotFoundError errno.ENOENT 
                                (os.strerror errno.ENOENT) 
                                pdk-path)))
    (unless (or ckt-path (not (os.path.exists ckt-path)))
      (raise (FileNotFoundError errno.ENOENT 
                                (os.strerror errno.ENOENT) 
                                ckt-path)))

    ;; Initialize parent Environment.
    (.__init__ (super ST1Env self) 
               [pdk-path] ckt-path
               max-moves random-start random-target tolerance
               :data-log-prefix data-log-prefix
               #_/ )

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- np.inf) :high np.inf
                                      :shape (, 12)   :dtype np.float32)))
  
  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wn0 Wn1 Wn2 Wp0 Wp1 Wp2) (unscale-value action 
                                                     self.action-scale-min 
                                                     self.action-scale-max)
          
          sizing {"Wn0" Wn0 "Wn1" Wn1 "Wn2" Wn2 
                  "Wp0" Wp0 "Wp1" Wp1 "Wp2" Wp2}]

      (.size-step (super) sizing)))
  
  (defn render [self &optional [mode "ascii"]]
    """
    Prints an ASCII Schematic of the Miller Amplifier courtesy
    https://github.com/Blokkendoos/AACircuit
    """
    (cond [(= mode "ascii")
           (print f"
  VDD #-------------o------------------.                                
                    |                  |                                
                 ||-+ MP0              |                                
                 ||->                  |                                
            .----||-+                  |                                
            |       |                  |                                
            |       |          MP2     |                                
            |       o--------+^+-----. |                                
            |       |        |||     | |                                
            |       |        ===     | |                                
            |    ||-+ MP1      |     | |                                
            |    ||->          |     | |                                
            o----||-+          |     | |                                
            |       |          |     | |                                
            |       |          |     | |                                
    I #-----o       o---------o------)-)-----# O                        
            |       |        |       | |                                
            |       |        |       | |                                
            |    ||-+ MN1    |       | |                                
            |    ||<-        |       | |                                
            o----||-+        |       | |                                
            |       |        ===     | |                                
            |       |        |^|     | |                                
            |       o--------+|+-----)-'                                
            |       |          MN2   |                                  
            |       |                |                                  
            |    ||-+ MN0            |                                  
            |    ||<-                |                                  
            '----||-+                |                                  
                    |                |                                  
  VSS #-------------o----------------'       
  " )]
          [True (.render (super) mode)])))

(defclass ST1XH035GeomEnv [ST1Env]
  """
  Schmitt Trigger.
  """

  (setv metadata {"render.modes" ["human" "ascii"]
                  "ace.env" "st1"})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-start True]
                                 ^bool [random-target True]
                                 ^float [tolerance 1e-3]
                                 ^str [data-log-prefix ""]]

    (.__init__ (super ST1XH035GeomEnv self) :pdk-path pdk-path 
                                            :ckt-path ckt-path
                                            :max-moves max-moves 
                                            :random-start random-start
                                            :data-log-prefix data-log-prefix
                                            #_/ )

    ;; The action space consists of 6 parameters ∈ [-1;1]. Each width of the
    ;; schmitt trigger: ["Wp0" "Wn0" "Wp2" "Wp1" "Wn2" "Wn1"]
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 6) 
                                 :dtype np.float32)
          self.action-scale-min (np.array (list (repeat 0.4e-6 6)))
          self.action-scale-max (np.array (list (repeat 150e-6 6)))))

(defn target-specification ^dict [self &optional ^bool [random False] 
                                                 ^bool [noisy True]]
    """
    Generate a noisy target specification.
    """
    (let [factor (if random (np.abs (np.random.normal 1 0.5)) 1.0)
          noise  (np.random.normal 1 0.01)
          delta  (if random (np.random.uniform :low 0.3 :high 0.5) 0.4)
          ts {"v_il"  (- (/ self.vdd 2.0) delta)
              "v_ih"  (+ (/ self.vdd 2.0) delta)
              "t_plh" (* 0.8e-9 factor)
              "t_phl" (* 0.8e-9 factor)
              #_/ }]
      (dfor (, p v) (.items ts)
        [ p (if noisy (* v noise) v) ]))))
