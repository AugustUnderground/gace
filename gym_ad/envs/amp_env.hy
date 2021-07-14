(import [itertools [product]])

(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])
(import [joblib :as jl])
(import [scipy [interpolate]])
(import [matplotlib.pyplot :as plt])

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

(defclass PrimitiveDevice []
  (defn __init__ [self prefix params-x params-y]
    (setv self.prefix prefix
          self.params-x params-x
          self.params-y params-y)

    (setv self.model (pt.jit.load f"{self.prefix}.pt")
          self.scale-x (jl.load f"{self.prefix}.X")
          self.scale-y (jl.load f"{self.prefix}.Y"))
    
    (-> self.model (.cpu) (.eval)))

  (defn predict [self X]
    (with [_ (pt.no-grad)]
      (let [_ (setv X.fug (np.log10 X.fug.values))
            X′ (-> X (get self.params-x) (. values) (self.scale-x.transform))
            Y′ (-> X′ (np.float32) (pt.from-numpy) (self.model) (.numpy))
            Y  (pd.DataFrame (self.scale-y.inverse-transform Y′)
                             :columns self.params-y)]
        (setv Y.jd (np.power 10 Y.jd.values))
        (setv Y.gdsw (np.power 10 Y.gdsw.values))
        Y))))

(defn ac-testbench [lib-path vdd vin iref cl]
  (setv spice-lib (SpiceLibrary lib-path))
  (setv tb (Circuit "ac_testbench"))
  (tb.include (get spice-lib "nmos"))
  (tb.I "bias" 0 "R" (u-A iref))
  (tb.VoltageSource "dd" "DD" 0 (u-V vdd))
  (tb.C "L" "O" 0 (u-F cl))
  (tb.VoltageSource "ip" "P" 0 (u-V vin))
  (tb.SinusoidalVoltageSource "in" "N" "E" 
                              :dc-offset (u-V 0.0) 
                              :ac-magnitude (u-V -1.0))
  (tb.VoltageControlledVoltageSource "in" "E" 0 "O" 0 (u-V 1.0))
  tb)

(defn dc-testbench [lib-path vdd vin vout iref cl]
  (setv spice-lib (SpiceLibrary lib-path))
  (setv tb (Circuit "dc_testbench"))
  (tb.include (get spice-lib "nmos"))
  (tb.I "bias" 0 "R" (u-A iref))
  (tb.VoltageSource "dd" "DD" 0 (u-V vdd))
  (tb.C "L" "O" 0 (u-F cl))
  (tb.VoltageSource "ip" "P" 0 (u-V vin))
  (tb.VoltageSource "in" "N" 0 (u-V vin))
  (tb.VoltageSource "out" "O" 0 (u-V vout))
  tb)

(defclass AmplifierEnv [gym.Env]
  (setv metadata {"render.modes" ["human" "ac" "dc" "bode"]})

  (defn __init__ [self amplifier lib-path vdd vi-cm vo-cm i-ref cl]
    (.__init__ (super AmplifierEnv self))
    (setv self.lib-path lib-path)

    (setv self.ac-tb (ac-testbench self.lib-path vdd vi-cm i-ref cl))
    (self.ac-tb.subcircuit amplifier)
    (self.ac-tb.X "OP" amplifier.NAME "R" "P" "N" "O" 0 "DD")
    (setv self.ac-amplifier (first self.ac-tb.subcircuits))

    (setv self.dc-tb (dc-testbench self.lib-path vdd vi-cm vo-cm i-ref cl))
    (self.dc-tb.subcircuit amplifier)
    (self.dc-tb.X "OP" amplifier.NAME "R" "P" "N" "O" 0 "DD")
    (setv self.dc-amplifier (first self.dc-tb.subcircuits)
          self.devices (list (filter #%(-> %1 (.upper) (.startswith "M")) 
                                     (list self.dc-amplifier.element-names)))
          self.num-devices (len self.devices))

    (setv self.op-params ["w" "l" "vdsat" "id" "gds" "gm" "gmbs" "cgg"]
          self.po-params ["fug" "gmoverid" "self_gain" "jd"]
          self.ac-params ["A0dB" "f3dB" "fug" "PM" "GM"]))
  
  (defn render [self &optional [mode "human"]]
    (cond [(= mode "ac")
           (print self.ac-tb)]
          [(= mode "dc")
           (print self.dc-tb)]
          [(= mode "human")
           (self.render :mode "ac")
           (self.render :mode "dc")]
          [(= mode "bode")
           (let [(, fig (, ax1 ax2)) (plt.subplots 2 1 :figsize (, 10 10))]
            (ax1.plot self.freq self.gain :label "Simulated Gain")
            (ax1.set-title "Loop Gain")
            (ax1.set-xscale "log")
            (ax1.set-ylabel "Gain [dB]")
            (ax1.grid "on")

            (ax2.plot self.freq self.phase :label "Simulated Phase")
            (ax2.set-title "Phase")
            (ax2.set-xscale "log")
            (ax2.set-xlabel "Frequency [Hz]")
            (ax2.set-ylabel "Phase [deg]")
            (ax2.grid "on")

            (plt.show))]
          [True
           (raise (NotImplementedError f"Mode {mode} not implented. Use 'human'."))]))

  (defn close [self]
    ; TODO FIXME DESTROY NETLIST / SIMULATOR / PYSPICE OBJECT
    None)

  (defn reset [self &optional [temp 27.0]]
    (setv self.ac-simulator (self.ac-tb.simulator :simulator "ngspice-subprocess"
                                                  :temperature temp
                                                  :nominal-temperature temp))

    (setv self.dc-simulator (self.dc-tb.simulator :temperature temp
                                                  :nominal-temperature temp))

    (setv self.moves 0)
    (first (self.feedback)))

  (defn op-simulation [self]
    (let [save-params (lfor (, d p) (product self.devices self.op-params) 
                            (.format "@M.XOP.{}[{}]" (.upper d) (.lower p)))
          _ (self.dc-simulator.save-internal-parameters #* save-params)
          _ (logging.disable logging.FATAL)
          op-analysis (self.dc-simulator.operating-point)
          op-data (dfor (, d p) (filter #%(not-in "cgg" %1) 
                                        (product self.devices self.op-params))
                        [(.format "{}:{}" d p) 
                         (->> p (.format "@M.XOP.{}[{}]" (.upper d) (.lower p)) 
                                (get op-analysis) 
                                (.item))])
          _ (for [d self.devices]
              (let [gm   (.item (get op-analysis (.format "@m.xop.{}[gm]" d)))
                    cgg  (.item (get op-analysis (.format "@m.xop.{}[cgg]" d)))
                    ids  (.item (get op-analysis (.format "@m.xop.{}[id]" d)))
                    gds  (.item (get op-analysis (.format "@m.xop.{}[gds]" d)))
                    w    (.item (get op-analysis (.format "@m.xop.{}[W]" d)))]
                (setv (get op-data (.format "{}:fug" d )) (/ gm (* 2.0 np.pi cgg)))
                (setv (get op-data (.format "{}:gmoverid" d )) (/ gm ids))
                (setv (get op-data (.format "{}:self_gain" d )) (/ gm gds))
                (setv (get op-data (.format "{}:jd" d )) (/ ids w))))
          _ (logging.disable logging.NOTSET)]
      op-data))

  (defn ac-simulation [self]
    (let [_ (logging.disable logging.FATAL)
          ac-analysis (self.ac-simulator.ac :start-frequency (u-Hz self.freq-start) 
                                            :stop-frequency  (u-Hz self.freq-stop) 
                                            :number-of-points 10
                                            :variation "dec")

          freq (-> ac-analysis (. frequency) (np.array))
          gain (- (-> ac-analysis (get "O") (np.absolute) (np.log10) (* 20))
                  (-> ac-analysis (get "N") (np.absolute) (np.log10) (* 20)))
          phase (- (-> ac-analysis (get "O") (np.angle :deg True))
                   (-> ac-analysis (get "N") (np.angle :deg True)))
          
          gf ((juxt #%(get gain %1) #%(get freq %1)) (np.argsort gain))
          pf ((juxt #%(get phase %1) #%(get freq %1)) (np.argsort phase))

          A0dB (interpolate.pchip-interpolate freq gain [1.0])
          A3dB (- A0dB 3)
          f3dB (interpolate.pchip-interpolate #* gf [A3dB])
          fug  (if (> A0dB 0) 
                   (interpolate.pchip-interpolate #* gf [0.0])
                   (np.array self.freq-stop))
          fp0  (interpolate.pchip-interpolate #* pf [0.0])
          PM   (if (> A0dB 0)
                   (interpolate.pchip-interpolate freq phase [fug])
                   (np.zeros 1))
          GM   (interpolate.pchip-interpolate freq gain [fp0]) 

          ac-data { "freq"  freq
                    "gain"  gain
                    "phase" phase 
                    "A0dB"  (-> A0dB (.flatten) (first))
                    "A3dB"  (-> A3dB (.flatten) (first))
                    "f3dB"  (-> f3dB (.flatten) (first))
                    "fug"   (-> fug  (.flatten) (first))
                    "fp0"   (-> fp0  (.flatten) (first))
                    "PM"    (-> PM   (.flatten) (first))
                    "GM"    (-> GM   (.flatten) (first)) }
          _ (logging.disable logging.NOTSET)]
      (setv self.freq freq
            self.gain gain
            self.phase phase)
      ac-data))

  (defn reward [self performance]
    (- (.sum (np.abs (- performance self.target)))))
  
  (defn feedback [self]
    (let [ac-dict (self.ac-simulation)
          op-dict (self.op-simulation) 
          op-vals (-> op-dict (.values) (np.fromiter :dtype np.float32))
          ac-vals (np.array (lfor p ["gain" "phase" "freq"] (get ac-dict p)))
          pp-vals (np.array (lfor p ["A0dB" "f3dB" "fug" "PM" "GM"] 
                                    (get ac-dict p)))
          obs (np.hstack (tuple (map #%(.flatten %1) [op-vals pp-vals ac-vals]))) 
          rew (self.reward pp-vals) ]
      (, obs rew)))

  (defn finished [self moves reward]
    (bool (or (> moves self.max-moves)
              (< (np.abs reward) self.target-tolerance))))

  (defn information [self]
    (let [op-params (lfor (, d p) (product self.devices self.op-params) 
                                  (.format "{}:{}" d p))
          po-params (lfor (, d p) (product self.devices self.po-params) 
                                  (.format "{}:{}" d p))]
      { "obskey" (flatten [ op-params 
                            po-params 
                            self.ac-params 
                            ["gain" "phase" "freq"]])})))
