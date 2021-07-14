(import [numpy :as np])
(import [pandas :as pd])
;(import [matplotlib.pyplot :as plt])

(import gym)
(import [gym.spaces [Box Discrete]])

(import [PySpice.Spice.Netlist [Circuit SubCircuitFactory]])
(import [PySpice.Spice.Library [SpiceLibrary]])
(import [PySpice.Unit [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defclass MillerAmpCkt [SubCircuitFactory]
  (setv NAME "miller"
        NODES (, 10 11 12 13 14 15)) ; REF INP INN OUT GND VDD
  (defn __init__ [self, C_C]
    (.__init__ (super))
    ; Biasing Current Mirror
    (self.MOSFET "NCM11" 10 10 14 14 :model "nmos")
    (self.MOSFET "NCM12" 16 10 14 14 :model "nmos")
    ; Differential Pair
    (self.MOSFET "ND11"  17 11 16 14 :model "nmos")
    (self.MOSFET "ND12"  18 12 16 14 :model "nmos")
    ;  PMOS Current Mirrors
    (self.MOSFET "PCM21" 17 17 15 15 :model "pmos")
    (self.MOSFET "PCM22" 18 17 15 15 :model "pmos")
    ; Output Stage
    (self.MOSFET "PCS"   13 18 15 15 :model "pmos")
    (self.MOSFET "NCM13" 13 10 14 14 :model "nmos")
    ; Compensation
    (self.C "c" 18 13 C_C@u_F)))

(defclass SymAmpEnv [AmplifierEnv]
  (setv metadata {"render.modes" ["human" "ac" "dc" "ascii"]})

  (defn __init__ [self &optional [nmos-prefix None] [pmos-prefix None] [lib-path None]
                                 [params-x ["gmid" "fug" "Vds" "Vbs"]]
                                 [params-y ["jd" "L" "gdsw" "Vgs"]]
                                 [max-moves 200]
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

    (setv self.params-x params-x
          self.params-y params-y
          self.nmos (PrimitiveDevice nmos-prefix self.params-x self.params-y)
          self.pmos (PrimitiveDevice pmos-prefix self.params-x self.params-y))

    (setv self.max-moves max-moves)

    (setv self.I-B0 I-B0
          self.V-DD vdd
          self.V-OCM vo-cm
          self.V-ICM vi-cm)

    (setv self.target target
          self.target-tolerance target-tolerance)

    (setv self.freq-start freq-start
          self.freq-stop  freq-stop)

    (setv self.op-amp (SymAmpCkt))
    (setv self.lib-path lib-path)

    (.__init__ (super SymAmpEnv self) self.op-amp self.lib-path vdd vi-cm vo-cm i-ref cl)

    (setv self.action-space (Box :low (np.float32 (np.concatenate (, (np.repeat gmid-lo 4) 
                                                                     (np.repeat fug-lo 4)
                                                                     (np.repeat M-lo 2))))
                                 :high (np.float32 (np.concatenate (, (np.repeat gmid-hi 4) 
                                                                      (np.repeat fug-hi 4)
                                                                      (np.repeat M-hi 2))))
                                 :dtype np.float32))

    (setv self.obs-dim (+ (* (len self.devices) 
                             (+ (len self.po-params) 
                                (- (len self.op-params) 1)))
                          (len self.ac-params))
          self.observation-spaces (Box :low (np.float32 (np.repeat (- np.inf) self.obs-dim))
                                       :high (np.float32 (np.repeat np.inf self.obs-dim))
                                       :dtype np.float32)))

  (defn render [self &optional [mode "ascii"]]
    (cond [(= mode "ascii")
           (print f"
o-------------o---------------o------------o--------------o----------o
              |               |            |              |           
              +-||         ||-+            +-||        ||-+           
              <-||         ||->            <-||        ||->           
              +-||----o----||-+            +-||----o---||-+           
              |       |       |            |       |      |           
              |       |       |            |       |      |           
              |       '-------o            o-------'      |           
              |               |            |              |           
              |               |            |              |           
              |               |            |              |           
              |            ||-+            +-||           |           
   o          |            ||<-            ->||           |           
   |          |      o-----||-+            +-||-----o     |           
   |          |               |            |              o-----o--o  
   |          |               '-----o------'              |     |     
   |          |                     |                     |     |     
   +-||       |                  ||-+                     |     |     
   ->||       |                  ||<-                     |     |     
   +-||-------)------------------||-+                     |     |     
   |          |                     |                     |    ---    
   |          |                     |                     |    ---    
   |          o-------.             |                     |     |     
   |          |       |             |                     |     |     
   |          |       |             |                     |     |     
   |          +-||    |             |                  ||-+     |     
   |          ->||    |             |                  ||<-     |     
   |          +-||----o-------------)------------------||-+     |     
   |          |                     |                     |     |     
   |          |                     |                     |     |     
   '----------o---------------------o---------------------o-----'     
                                    |                                 
                                   ===                                
                                   GND                                " )]
          [True (.render (super) mode)]))

  (defn electric2geometric [self gmid_ncm12 gmid_ndp12 gmid_pcm212 gmid_ncm32  
                            fug_ncm12  fug_ndp12  fug_pcm212  fug_ncm32
                            MN MP]
    (let [char {}
          V-X 0.2
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
    (setv (get action (slice 4 8)) (np.power 10 (get action (slice 4 8))))
    action)

  (defn step [self action]
    (let [scaled-action (self.scale-action action)
          sizing (self.electric2geometric #* scaled-action) 
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
    (, observation reward done info))))
