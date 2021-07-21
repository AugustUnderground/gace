(import os)
(import sys)
;(import multiprocess)
(import [functools [partial]])

(import [numpy :as np])
(import [pandas :as pd])

(import [skopt [gp-minimize]])

(import gym)
(import [gym.spaces [Box Discrete]])

(import [PySpice.Spice.Netlist [Circuit SubCircuitFactory]])
(import [PySpice.Spice.Library [SpiceLibrary]])
(import [PySpice.Unit [*]])

(import [.amp_env [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

;(multiprocess.set-executable (.replace sys.executable "hy" "python"))

(defclass SymAmpCkt [SubCircuitFactory]
  (setv NAME "symamp"
        NODES (, 10 11 12 13 14 15)) ; B INP INN OUT GND VDD
  (defn __init__ [self]
    (.__init__ (super))
    ; Biasing Current Mirror
    (self.MOSFET "NCM11"  10 10 14 14 :model "nmos")
    (self.MOSFET "NCM12"  16 10 14 14 :model "nmos")
    ; Differential Pair
    (self.MOSFET "ND11"   17 12 16 14 :model "nmos")
    (self.MOSFET "ND12"   18 11 16 14 :model "nmos")
    ; PMOS Current Mirrors
    (self.MOSFET "PCM221" 17 17 15 15 :model "pmos")
    (self.MOSFET "PCM222" 19 17 15 15 :model "pmos")
    (self.MOSFET "PCM211" 18 18 15 15 :model "pmos")
    (self.MOSFET "PCM212" 13 18 15 15 :model "pmos")
    ; NMOS Current Mirror
    (self.MOSFET "NCM31"  19 19 14 14 :model "nmos")
    (self.MOSFET "NCM32"  13 19 14 14 :model "nmos")))

(defclass SymAmpEnv [AmplifierEnv]
  (setv metadata {"render.modes" ["human" "ac" "dc" "ascii" "bode"]})

  (defn __init__ [self &optional [nmos-prefix None] [pmos-prefix None] [lib-path None]
                                 [params-x ["gmid" "fug" "Vds" "Vbs"]]
                                 [params-y ["jd" "L" "gdsw" "Vgs"]]
                                 [max-moves 200] [close-target True]
                                 [target [48.0 1e4 4e6 30 0]]
                                 [target-tolerance 1e-3]
                                 [I-B0 10e-6] [M-lo 0.1] [M-hi 10.0]
                                 [gmid-lo 5.0] [gmid-hi 15.0]
                                 [fug-lo 6.0] [fug-hi 11.0]
                                 [freq-start 1.0] [freq-stop 1e11] 
                                 [vdd 1.2] [vi-cm 0.6] [vo-cm 0.6] 
                                 [i-ref 10e-6] [cl 15e-12]]

    (if-not (and nmos-prefix pmos-prefix lib-path)
      (raise (TypeError f"SymAmpEnv requires 'nmos_prefix', 'pmos_prefix' and 'lib_path' kwargs.")))

    (setv self.last-reward -Inf
          self.reset-close-to-target close-target)

    (setv self.gmid-lo gmid-lo
          self.gmid-hi gmid-hi
          self.fug-lo fug-lo
          self.fug-hi fug-hi
          self.m-lo M-lo
          self.m-hi M-hi)

    (setv self.denorm-action 
          (fn [action-lo action-hi action]
            (+ (* (/ (+ action 1.0) 2.0) 
                  (- action-hi action-lo)) 
               action-lo)))

    (setv self.d-gmid (partial self.denorm-action gmid-lo gmid-hi)
          self.d-fug  (partial self.denorm-action fug-lo fug-hi)
          self.d-m    (partial self.denorm-action M-lo M-hi))

    (setv self.norm-action 
          (fn [action-lo action-hi action]
            (- (* 2.0 (/ (- action action-lo) 
                         (- action-hi action-lo))) 
               1.0)))

    (setv self.n-gmid (partial self.norm-action gmid-lo gmid-hi)
          self.n-fug  (partial self.norm-action fug-lo fug-hi)
          self.n-m    (partial self.norm-action M-lo M-hi))

    (setv self.params-x params-x
          self.params-y params-y
          self.nmos (PrimitiveDevice nmos-prefix self.params-x self.params-y)
          self.pmos (PrimitiveDevice pmos-prefix self.params-x self.params-y))

    (setv self.max-moves max-moves)

    (setv self.I-B0 I-B0
          self.V-DD vdd
          self.V-OCM vo-cm
          self.V-ICM vi-cm)

    (setv self.target (np.array target)
          self.target-tolerance target-tolerance)

    (setv self.freq-start freq-start
          self.freq-stop  freq-stop)

    (setv self.op-amp (SymAmpCkt))
    (setv self.lib-path lib-path)

    (.__init__ (super SymAmpEnv self) self.op-amp self.lib-path vdd vi-cm vo-cm i-ref cl)

    (setv self.num-gmid 4 self.num-fug 4 self.num-m 2
          self.act-dim  (+ self.num-gmid self.num-fug self.num-m)
          self.action-space (Box :low -1.0 
                                 :high 1.0 
                                 :shape (, self.act-dim)
                                 :dtype np.float32))

    (setv self.obs-dim 448
          self.observation-space (Box :low (np.float32 (np.repeat (- np.inf) self.obs-dim))
                                      :high (np.float32 (np.repeat np.inf self.obs-dim))
                                      :shape (, self.obs-dim)
                                      :dtype np.float32)))

  (defn reset [self &optional [target None] [init-act None] [temp 27]]
    (let [θ (if (> self.reset-counter 0)
                (np.log10 self.reset-counter)
                self.reset-counter)
          new-target (or target (* θ (np.random.rand (len self.target))))
          reset-obs (.reset (super) :temp temp)]
      (cond [(is-not init-act None) 
             (-> init-act (self.step) (first))]
            [self.reset-close-to-target 
             (-> (fn [act] (-> act (self.step) (second) (-)))
                 (gp-minimize [ #* (repeat (, -1.0 1.0) self.act-dim) ] 
                              :acq-func "PI"
                              :n-calls 15
                              :n-random-starts 7
                              :noise 1e-3
                              :random-state 666
                              :n-jobs 1)
                 (. x)
                 (self.step)
                 (first))]
            [True reset-obs])))

  (defn electric2geometric [self gmid_ncm12 gmid_ndp12 gmid_pcm212 gmid_ncm32  
                            fug_ncm12  fug_ndp12  fug_pcm212  fug_ncm32
                            MN MP]
    (let [char {}
          V-X 0.2
          MN 1
          MP 4
          I-B1 (* MN self.I-B0)
          I-B2 (* 0.5 I-B1 MP)
          input-pcm212 (pd.DataFrame (np.array [[gmid-pcm212 fug-pcm212 (- self.V-DD self.V-OCM) 0.0]])
                                     :columns self.params-x)
          input-ncm32  (pd.DataFrame (np.array [[gmid-ncm32 fug-ncm32 self.V-OCM 0.0]])
                                     :columns self.params-x)
          input-ncm12  (pd.DataFrame (np.array [[gmid-ncm12 fug-ncm12 V-X 0.0]])
                                     :columns self.params-x)]

      (setv (get char "MPCM212")      (.join (self.pmos.predict input-pcm212) input-pcm212)
            (-> char (get "MPCM212") 
                     (get "W"))       (/ I-B2 (-> char (get "MPCM212") (get "jd") (. values)) MP)
            (-> char (get "MPCM212") 
                     (get "M"))       MP
            (get char "MPCM211")      (.copy (get char "MPCM212"))
            (-> char (get "MPCM211") 
                     (get "M"))       1
            (get char "MPCM222")      (get char "MPCM212")
            (get char "MPCM221")      (get char "MPCM211"))

      (setv (get char "MNCM32")       (.join (self.nmos.predict input-ncm32) input-ncm32)
            (-> char (get "MNCM32") 
                     (get "W"))       (/ I-B2 (-> char (get "MNCM32") (get "jd") (. values)))
            (-> char (get "MNCM32") 
                     (get "M"))       1
            (get char "MNCM32")       (get char "MNCM32"))

      (setv V-GS (first (-> char (get "MNCM32") (get "Vgs") (. values))))
      (setv input-ndp12 (pd.DataFrame (np.array [[gmid-ndp12 fug-ndp12 (- self.V-DD V-GS V-X) (- V-X)]])
                                      :columns self.params-x))

      (setv (get char "MND12")        (.join (self.nmos.predict input-ndp12) input-ndp12)
            (-> char (get "MND12") 
                     (get "W"))       (/ (* 0.5 I-B1) (-> char (get "MND12") (get "jd") (. values)))
            (-> char (get "MND12") 
                     (get "M"))       1
            (get char "MND11")        (get char "MND12"))

      (setv (get char "MNCM12")       (.join (self.nmos.predict input-ncm12) input-ncm12)
            (-> char (get "MNCM12") 
                     (get "W"))       (/ I-B1 (-> char (get "MNCM12") (get "jd") (. values)) MN)
            (-> char (get "MNCM12") 
                     (get "M"))       MN
            (get char "MNCM11")       (.copy (get char "MNCM12"))
            (-> char (get "MNCM11") 
                     (get "M"))       1)
      char))

  (defn scale-action [self action]
    (let [gmids (self.n-gmid (get action (slice 0 self.num-gmid)))
          fugs  (self.n-fug (get action (slice self.num-gmid (+ self.num-gmid self.num-fug))))
          ms    (self.n-m (get action (slice (+ self.num-gmid self.num-fug) None)))]
      (.flatten (np.hstack [gmids fugs ms]))))

  (defn unscale-action [self action]
    (let [gmids (self.d-gmid (get action (slice 0 self.num-gmid)))
          fugs  (self.d-fug (get action (slice self.num-gmid (+ self.num-gmid self.num-fug))))
          ms    (self.d-m (get action (slice (+ self.num-gmid self.num-fug) None)))]
      (.flatten (np.hstack [gmids fugs ms]))))

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
