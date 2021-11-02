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

(import [.inv [Nand4Env]])
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

(defclass NAND4Env [Nand4Env]
  """
  Derived amplifier class, implementing the Miller Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  Observation Space:
    - See AmplifierXH035Env

  Action Space:
    Continuous Box a: (14,) ∈ [1.0;1.0]

    Where
    a = [ gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
          fug-cm1  fug-cm2  fug-cm3  fug-dp1 
          i1 i2 ] 

      where i1 and i2 are the branch currents.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-start True]
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
    (.__init__ (super NAND4Env self) 
               [pdk-path] ckt-path
               max-moves random-start tolerance
               :data-log-prefix data-log-prefix
               #_/ )

    ;; The `Box` type observation space consists of perforamnces, the distance
    ;; to the target, as well as general information about the current
    ;; operating point.
    (setv self.observation-space (Box :low (- self.vdd) :high self.vdd
                                      :shape (, 12)  :dtype np.float32)))
  
  (defn step [self action]
    """
    Takes an array of geometric parameters for each building block and mirror
    ratios This is passed to the parent class where the netlist ist modified
    and then simulated, returning observations, reward, done and info.
    """
    (let [(, Wn0 Wn1 Wn2 Wn3 Wp1) (unscale-value action 
                                                 self.action-scale-min 
                                                 self.action-scale-max)
          
          sizing {"wn0" Wn0 "wn1" Wn1 "wn2" Wn2 "wn3" Wn3 "wp" Wp1}]

      (.size-step (super) sizing)))
  
  (defn render [self &optional [mode "ascii"]]
    """
    Prints an ASCII Schematic of the Miller Amplifier courtesy
    https://github.com/Blokkendoos/AACircuit
    """
    (cond [(= mode "ascii")
           (print f"
VDD #---------o----------o----------o----------.                        
              |          |          |          |                        
              |          |          |          |                        
           ||-+ MP0   ||-+ MP1   ||-+ MP2   ||-+ MP3                    
           ||->       ||->       ||->       ||->                        
        .--||-+    .--||-+    .--||-+    .--||-+                        
        |     |    |     |    |     |    |     |                        
        |     '----)-----o----)-----o----)-----o----# O                 
        |          |          |          |     |                        
        |          |          |          |  ||-+ MN3                    
        |          |          |          |  ||<-                        
 I3 #---)----------)----------)----------o--||-+                        
        |          |          |                |                        
        |          |          |                |                        
        |          |          |                |                        
        |          |          |             ||-+ MN2                    
        |          |          |             ||<-                        
 I2 #---)----------)----------o-------------||-+                        
        |          |                           |                        
        |          |                           |                        
        |          |                           |                        
        |          |                        ||-+ MN1                    
        |          |                        ||<-                        
 I1 #---)----------o------------------------||-+                        
        |                                      |                        
        |                                      |                        
        |                                      |                        
        |                                   ||-+ MN0                    
        |                                   ||<-                        
 I0 #---o-----------------------------------||-+                        
                                               |                        
                                               |                        
VSS #------------------------------------------'   
  " )]
          [True (.render (super) mode)])))

(defclass NAND4XH035GeomEnv [NAND4Env]
  """
  4 NAND Gate Inverter Chain.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [pdk-path None] ^str [ckt-path None] 
                                 ^int [max-moves 200]
                                 ^bool [random-start True]
                                 ^float [tolerance 1e-3]
                                 ^str [data-log-prefix ""]]

    ;; Supply Voltage
    (setv self.vdd 3.3)

    (.__init__ (super NAND4XH035GeomEnv self) :pdk-path pdk-path 
                                              :ckt-path ckt-path
                                              :max-moves max-moves 
                                              :random-start random-start
                                              :data-log-prefix data-log-prefix
                                              #_/ )

    ;; The action space consists of 5 parameters ∈ [-1;1]. Each width of the
    ;; inverter chain:  ['wn0', 'wp', 'wn2', 'wn1', 'wn3']
    (setv self.action-space (Box :low -1.0 :high 1.0 
                                 :shape (, 5) 
                                 :dtype np.float32)
          self.action-scale-min (np.array (list (repeat 0.4e-6 5)))
          self.action-scale-max (np.array (list (repeat 150e-6 5))))))
