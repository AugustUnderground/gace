(import os)
(import sys)
(import [functools [partial]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(import [skopt [gp-minimize]])

(import gym)
(import [gym.spaces [Dict Box Discrete Tuple]])

(import [.amp_env [*]])
(import [.prim_dev [*]])

(import jpype)
(import jpype.imports)
(import [jpype.types [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

;; THIS WILL BE FIXED IN HY 1.0!
;(import multiprocess)
;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass SymAmpXH035Env [AmplifierEnv]
  """
  Derived amplifier class, implementing the Symmetrical Amplifier in the XFAB
  XH035 Technology. Only works in combinatoin with the right netlists.
  """

  (setv metadata {"render.modes" ["human" "ascii"]})

  (defn __init__ [self &optional ^str [nmos-path None]  ^str [pmos-path None] 
                                 ^str [lib-path None]   ^str [jar-path None]
                                 ^str [sim-path "/tmp"] ^str [ckt-path None]
                                 ^int [max-moves 200]   ^bool [close-target True]
                                 ^float [target-tolerance 1e-3] 
                                 ^dict [target None]]
    """
    Constructs a Symmetrical Amplifier Environment with XH035 device models and
    the corresponding netlist.
    Arguments:
      nmos-path:  Prefix path, expects to find `nmos-path/model.pt`, 
                  `nmos-path/scale.X` and `nmos-path/scale.Y` at this location.
      pmos-path:  Same as 'nmos-path', but for PMOS model.

      lib-path:   Path to XFAB MOS devices for XH035 pdk, something like 
                  /path/to/pdk/tech/xh035/cadence/vX_X/spectre/vX_X_X/mos
      jar-path:   Path to edlab.eda.characterization jar, something like
                  $HOME/.m2/repository/edlab/eda/characterization/$VERSION/characterization-$VERSION-jar-with-dependencies.jar
                  Has to be 'with-dependencies' otherwise waveforms can't be
                  loaded etc.
      sim-path:   Path to where the simulation results will be stored. Default
                  is /tmp.
      ckt-path:   Path where the amplifier netlist and testbenches are located.
      
      max-moves:  Maximum amount of steps the agent is allowed to take per
                  episode, before it counts as failed. Default = 200.
      
      close-target: If True (default), on each reset, a random target will be
                    chosen and by bayesian optimization, a location close to it
                    will be found for the starting point of the agent. This
                    increases the reset time significantly.
      target-tolerance: (| target - performance | <= tolerance) ? Success.
      target: Specific target, if given, no random targets will be generated,
              and the agent tries to find the same one over and over again.
    """

    (if-not (and lib-path jar-path ckt-path)
      (raise (TypeError f"SymAmpEnv requires 'lib_path', 'ckt-path' and 'jar-path' kwargs.")))

    ;; Launch JVM and import the Corresponding Amplifier Characterization Library
    ; f"/home/ynk/.m2/repository/edlab/eda/characterization/0.0.1/characterization-0.0.1-jar-with-dependencies.jar"
    (jpype.startJVM :classpath jar-path)
    (import [edlab.eda.characterization [Opamp2XH035Characterization]])
    
    ;; Load the PyTorch NMOS/PMOS Models for converting paramters.
    (setv self.nmos (XFAB f"{nmos-path}/model.pt" f"{nmos-path}/scale.X" f"{nmos-path}/scale.Y")
          self.pmos (XFAB f"{pmos-path}/model.pt" f"{pmos-path}/scale.X" f"{pmos-path}/scale.Y"))

    ;; Specify constants as they are defined in the netlist. 
    (setv self.vs   0.5
          self.cl   5e-12
          self.rl   100e6
          self.i0   3e-6
          self.vsup 3.3
          self.fin  1e3
          self.dev  1e-4)

    ;; Initialize parent Environment.
    (.__init__ (super SymAmpEnv self) Opamp2XH035Characterization
                                      lib-path sim-path ckt-path 
                                      params-x params-y max-moves 
                                      target-tolerance)

    ;; Generate random target of None was provided.
    (setv self.same-target  (bool target)
          self.target       (or target (self.random-target)))

    ;; Specify geometric and electric parameters, these have to align with the
    ;; parameters defined in the netlist.
    (setv self.geometric-parameters [ "Lcm1" "Lcm2" "Lcm3" "Ld" "Mcm11" "Mcm12"
                                      "Mcm21" "Mcm22" "Mcm31" "Mcm32" "Md"
                                      "Wcm1" "Wcm2" "Wcm3" "Wd" ]
          self.electric-parameters [ "gmid_cm1" "gmid_cm2" "gmid_cm3" "gmid_dp1" 
                                     "fug_cm1" "fug_cm2" "fug_cm3" "fug_dp1" ])

    ;; The action space consists of 8 parameters ∈ [0;1]. One gm/id and fug for
    ;; each building block. This is subject to change and will include branch
    ;; currents / mirror ratios in the future.
    (setv self.action-space (Box :low -1.0 :high 1.0 :shape (, 8) 
                                 :dtype np.float32)))

  (defn electric2geometric [ self gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
                                  fug-cm1  fug-cm2  fug-cm3  fug-dp1 ]
    """
    Takes electric parameters for each building block in the circuit and
    transforms them into corresponding geometric parameters with previosly
    trained machine learning models.
    """
    (let [vx 1.25 mn 1 mp 4
          i1 (* mn self.i0)
          i2 (* 0.5 i1 mp)
          (, jd l gdsw vgs vdsat gmbsw) (self.nmos (np.array [[gmid-cm1 fug-cm1 0.5 1.0]]))

          (, ) (self.pmos (np.array [[gmid-cm2 fug-cm2 0.5 1.0]]))
          (, ) (self.nmos (np.array [[gmid-cm3 fug-cm3 0.5 1.0]]))
          (, ) (self.nmos (np.array [[gmid-dp1 fug-dp1 0.5 1.0]]))
          ]
      {

      }
    ))

  (defn step [self action]
    (let [real-action (self.unscale-action (np.array action))
          sizing (self.electric2geometric #* real-action) 
          sizing-data (pd.concat (.values sizing) :names (.keys sizing))
          _ (setv sizing-data.index (.keys sizing))

          _ (for [device sizing-data.index]
              (setv (-> self.dc-amplifier (.element device) (. width)) 
                    (-> sizing-data (. loc) (get device) (. W))
                    (-> self.dc-amplifier (.element device) (. length)) 
                    (-> sizing-data (. loc) (get device) (. L))
                    (-> self.dc-amplifier (.element device) (. multiplier)) 
                    (-> sizing-data (. loc) (get device) (. M)))
              (setv (-> self.ac-amplifier (.element device) (. width)) 
                    (-> sizing-data (. loc) (get device) (. W))
                    (-> self.ac-amplifier (.element device) (. length)) 
                    (-> sizing-data (. loc) (get device) (. L))
                    (-> self.ac-amplifier (.element device) (. multiplier)) 
                    (-> sizing-data (. loc) (get device) (. M))))
          _ (setv self.moves (inc self.moves))
          (, observation 
             reward ) (self.feedback)
          done        (self.finished self.moves reward)
          info        (self.information) ]
    (setv self.last-reward reward)
    (, observation reward done info)))

  (defn render [self &optional [mode "ascii"]]
    (cond [(= mode "ascii")
           (print f"
o-------------o---------------o------------o--------------o----------o VDD
              |               |            |              |           
      MPCM222 +-||         ||-+ MPCM221    +-||        ||-+ MPCM212
              <-||         ||->            <-||        ||->           
              +-||----o----||-+    MPCM211 +-||----o---||-+           
              |       |       |            |       |      |           
              |       |       |            |       |      |           
              |       '-------o            o-------'      |           
              |               |            |              |           
              |               |            |              |           
              |               |            |              |           
 Iref         |            ||-+ MND11      +-||           |           
   o          |            ||<-            ->||           |           
   |          |  VI+ o-----||-+      MND12 +-||-----o VI- |           
   |          |               |     X      |              o-----o--o VO
   |          |               '-----o------'              |     |     
   |          |                     |                     |     |     
   +-|| MNCM11|           MNCM12 ||-+                     |     |     
   ->||       |                  ||<-                     |     |     
   +-||-------)------------------||-+                     |     |     
   |          |                     |                     |    --- CL
   |          |                     |                     |    ---    
   |          o-------.             |                     |     |     
   |          |       |             |                     |     |     
   |          |       |             |                     |     |     
   |   MNCM31 +-||    |             |           MNCM32 ||-+     |     
   |          ->||    |             |                  ||<-     |     
   |          +-||----o-------------)------------------||-+     |     
   |          |                     |                     |     |     
   |          |                     |                     |     |     
   '----------o---------------------o---------------------o-----'     
                                    |                                 
                                   ===                                
                                   VSS                                " )]
          [True (.render (super) mode)])))
