(import os)
(import errno)
(import [numpy :as np])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defn target-specification [^str ace-id ^str ace-backend
                  &optional ^bool [random False] 
                            ^bool [noisy True]]
  (let [ts (cond [(and (in ace-id ["op1" "op6"]) (= ace-backend "xh035-3V3"))
                  {"a_0"          105.0
                   "ugbw"         3500000.0
                   "pm"           110.0
                   "gm"           -45.0
                   "sr_r"         2700000.0
                   "sr_f"         -2700000.0
                   "vn_1Hz"       6.0e-06
                   "vn_10Hz"      2.0e-06
                   "vn_100Hz"     6.0e-07
                   "vn_1kHz"      1.5e-07
                   "vn_10kHz"     5.0e-08
                   "vn_100kHz"    2.6e-08
                   "psrr_n"       120.0
                   "psrr_p"       120.0
                   "cmrr"         110.0
                   "v_il"         0.7
                   "v_ih"         3.2
                   "v_ol"         0.1
                   "v_oh"         3.2
                   "i_out_min"    -7e-5
                   "i_out_max"    7e-5
                   "overshoot_r"  0.0005
                   "overshoot_f"  0.0005
                   "voff_stat"    0.003
                   "voff_sys"     -2.5e-05
                   "A"            5.0e-09
                   #_/ }]
                 ;[(and (in ace-id ["op2" "op3"]) (= ace-backend "xh035-3V3"))
                 [(in ace-id ["op2" "op3"])
                  {"a_0"         55.0
                   "ugbw"        3750000.0
                   "pm"          65.0
                   "gm"          -30.0
                   "sr_r"        3750000.0
                   "sr_f"        -3750000.0
                   "vn_1Hz"      5e-06
                   "vn_10Hz"     2e-06
                   "vn_100Hz"    5e-07
                   "vn_1kHz"     1.5e-07
                   "vn_10kHz"    5e-08
                   "vn_100kHz"   2.5e-08
                   "psrr_n"      80.0
                   "psrr_p"      80.0
                   "cmrr"        80.0
                   "v_il"        0.9
                   "v_ih"        3.2
                   "v_ol"        1.65
                   "v_oh"        3.2
                   "i_out_min"   -7e-5
                   "i_out_max"   7e-5
                   "overshoot_r" 2.0
                   "overshoot_f" 2.0
                   "voff_stat"   3e-3
                   "voff_sys"    -1.5e-3
                   "A"           5.5e-10
                   #_/ }]
                 [(and (in ace-id ["op4" "op5"]) (= ace-backend "xh035-3V3"))
                  {"a_0"         55.0
                   "ugbw"        3750000.0
                   "pm"          65.0
                   "gm"          -30.0
                   "sr_r"        3750000.0
                   "sr_f"        -3750000.0
                   "vn_1Hz"      5e-06
                   "vn_10Hz"     2e-06
                   "vn_100Hz"    5e-07
                   "vn_1kHz"     1.5e-07
                   "vn_10kHz"    5e-08
                   "vn_100kHz"   2.5e-08
                   "psrr_n"      80.0
                   "psrr_p"      80.0
                   "cmrr"        80.0
                   "v_il"        0.9
                   "v_ih"        3.2
                   "v_ol"        1.65
                   "v_oh"        3.2
                   "i_out_min"   -7e-5
                   "i_out_max"   7e-5
                   "overshoot_r" 2.0
                   "overshoot_f" 2.0
                   "voff_stat"   3e-3
                   "voff_sys"    -1.5e-3
                   "A"           5.5e-10
                   #_/ }]
                 [(and (= ace-id "nand4") (= ace-backend "xh035-3V3"))
                  {"vs0" 1.65
                   "vs1" 1.65
                   "vs2" 1.65
                   "vs3" 1.65
                   #_/ }]
                 [(and (= ace-id "nand4") (= ace-backend "sky130-1V8"))
                  {"vs0" 0.9
                   "vs1" 0.9
                   "vs2" 0.9
                   "vs3" 0.9
                   #_/ }]
                 [(and (= ace-id "st1") (= ace-backend "xh035-3V3"))
                  {"v_il"  (- 1.65 0.4) 
                   "v_ih"  (+ 1.65 0.4)
                   "t_plh" 0.8e-9
                   "t_phl" 0.8e-9
                   #_/ }]
                 [True (raise (NotImplementedError errno.ENOSYS
                            (os.strerror errno.ENOSYS) 
                            (.format "There is no target for {} in {}"
                                     ace-id ace-backend))) ])
        factor (cond [random (np.abs (np.random.normal 1 0.5))]
                     [noisy  (np.random.normal 1 0.01)]
                     [True   1.0])]
    (dfor (, p v) (.items ts)
        [ p (if noisy (* v factor) v) ])))

(defn reward-condition ^dict [^str ace-id &optional ^float [tolerance 1e-3]]
  (setv == (fn [a b] (<= (/ (np.abs (- a b)) b) tolerance)))
  (cond [(.startswith ace-id "op")
         {"a_0"         <=
          "ugbw"        <=
          "pm"          <=
          "gm"          >=
          "sr_r"        <=
          "sr_f"        >=
          "vn_1Hz"      >=
          "vn_10Hz"     >=
          "vn_100Hz"    >=
          "vn_1kHz"     >=
          "vn_10kHz"    >=
          "vn_100kHz"   >=
          "psrr_n"      <=
          "psrr_p"      <=
          "cmrr"        <=
          "v_il"        >=
          "v_ih"        <=
          "v_ol"        >=
          "v_oh"        <=
          "i_out_min"   >=
          "i_out_max"   >=
          "overshoot_r" >=
          "overshoot_f" >=
          "voff_stat"   >=
          "voff_sys"    <=
          "A"           >=
          #_/ }]
        [(.startswith ace-id "nand")
         {"vs0" ==
          "vs1" ==
          "vs2" ==
          "vs3" ==
          #_/ }]
        [(.startswith ace-id "st")
         {"v_il"  ==
          "v_ih"  ==
          "t_plh" ==
          "t_phl" ==
         #_/ }]
        [True (raise (NotImplementedError errno.ENOSYS
                      (os.strerror errno.ENOSYS) 
                      (.format "There is no reward condition for {} in {}"
                               ace-id ace-backend)))]))
