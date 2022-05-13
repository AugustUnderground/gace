(import os)
(import errno)

(import [numpy :as np])

(import [.func [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])
(import [hy.contrib.pprint [pp pprint]])

(defn performance-scaler [ace ^str ace-id ^str ace-backend]
  (let [vdd (get (design-constraints ace ace-id ace-backend) "vsup" "init")]
    (cond [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) 
                (in ace-id ["op1" "op6"]))
           {"a_0"         [25.0 120.0]  ; 100.0
            "ugbw"        [2.5 8.0]     ; 2500000.0
            "pm"          [0.0 150.0]   ; 80.0
            "gm"          [0.0 130.0]   ; -45.0
            "sr_r"        [0.0 8.0]     ; 2500000.0
            "sr_f"        [0.0 8.0]     ; -2500000.0
            "vn_1Hz"      [-8.0 -4.0]   ; 6.0e-06
            "vn_10Hz"     [-8.0 -4.0]   ; 2.0e-06
            "vn_100Hz"    [-8.0 -4.0]   ; 6.0e-07
            "vn_1kHz"     [-8.0 -4.0]   ; 1.5e-07
            "vn_10kHz"    [-8.0 -4.0]   ; 5.0e-08
            "vn_100kHz"   [-8.0 -4.0]   ; 2.6e-08
            "psrr_n"      [30.0 170.0]  ; 80.0
            "psrr_p"      [70.0 150.0]  ; 80.0
            "cmrr"        [70.0 150.0]  ; 80.0
            "v_il"        [0.0 vdd]     ; 0.9
            "v_ih"        [0.0 vdd]     ; 2.7
            "v_ol"        [0.0 vdd]     ; 0.1
            "v_oh"        [0.0 vdd]     ; 2.7
            "i_out_min"   [-6.0 -1.0]   ; -7e-5
            "i_out_max"   [-6.0 -1.0]   ; 7e-5
            "overshoot_r" [0.0 130.0] ; 2.0
            "overshoot_f" [0.0 130.0] ; 2.0
            "cof"         [5.0 9.0] ;  48000000.0
            "voff_stat"   [-4.0 0.0] ; 0.003
            "voff_sys"    [-7.0 0.0] ; -0.003
            "A"           [-6.5 -5.5] ; 5.5e-09
            #_/ }]
          [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) 
                (in ace-id ["op2" "op3"]))
           {"a_0"         [25.0 70.0] ; 55.0
            "ugbw"        [5.0 10.0]  ; 3500000.0
            "pm"          [0.0 120.0] ; 65.0
            "gm"          [0.0 80.0] ; -20.0
            "sr_r"        [4.0 8.0] ; 3500000.0
            "sr_f"        [4.0 8.0] ; -3500000.0
            "vn_1Hz"      [-10.0 -4.0] ; 5e-06
            "vn_10Hz"     [-10.0 -4.0] ; 2e-06
            "vn_100Hz"    [-10.0 -4.0] ; 5e-07
            "vn_1kHz"     [-10.0 -4.0] ; 1.5e-07
            "vn_10kHz"    [-10.0 -4.0] ; 5e-08
            "vn_100kHz"   [-10.0 -4.0] ; 2.5e-08
            "cmrr"        [70 150.0] ; 80.0
            "psrr_n"      [30.0 70.0] ; 60.0
            "psrr_p"      [40.0 140.0] ; 80.0
            "v_il"        [0.0 (* 2.0 vdd)] ; (* vdd 0.25) ; 0.9
            "v_ih"        [0.0 (* 2.0 vdd)] ; (* vdd 0.95) ; 3.2
            "v_ol"        [0.0 (* 2.0 vdd)] ; (* vdd 0.50) ; 1.65
            "v_oh"        [0.0 (* 2.0 vdd)] ; (* vdd 0.95) ; 3.2
            "i_out_min"   [-6.0 -3.0] ; -7e-5
            "i_out_max"   [-6.0 -3.0] ; 7e-5
            "overshoot_r" [0.0 130.0] ; 2.0
            "overshoot_f" [0.0 130.0] ; 2.0
            "cof"         [5.0 9.0] ;  48000000.0
            "voff_stat"   [-5.0 0.0] ; 0.003
            "voff_sys"    [-5.0 0.0] ; -0.003
            "A"           [-15.0 -5.0] ; 5.5e-10
            #_/ }]
          [(and (in ace-backend ["xh035-3V3" "xh018-1V8"]) 
                (in ace-id ["op8" "op9"]))
           {"a_0"         [-100.0 100.0]  ; 70.0
            "ugbw"        [2.0 7.0]   ; 950000.0
            "pm"          [0.0 150.0]  ; 85.0
            "gm"          [0.0 100.0]  ; -70.0
            "sr_r"        [0.0 8.0]    ; 880000.0
            "sr_f"        [0.0 8.0]    ; -880000.0
            "vn_1Hz"      [-10.0 -1.0] ; 5e-06
            "vn_10Hz"     [-10.0 -1.0] ; 2e-06
            "vn_100Hz"    [-10.0 -1.0] ; 5e-07
            "vn_1kHz"     [-10.0 -1.0] ; 1.5e-07
            "vn_10kHz"    [-10.0 -1.0] ; 5e-08
            "vn_100kHz"   [-10.0 -1.0] ; 2.5e-08
            "cmrr"        [-5.0 200.0] ; 120
            "psrr_n"      [-60.0 160.0] ; 90.0
            "psrr_p"      [-60.0 180.0] ; 150.0
            "v_il"        [0.0 (* 2.0 vdd)] ; (* vdd 0.25) ; 0.9
            "v_ih"        [0.0 (* 2.0 vdd)] ; (* vdd 0.95) ; 3.2
            "v_ol"        [0.0 (* 2.0 vdd)] ; (* vdd 0.50) ; 1.65
            "v_oh"        [0.0 (* 2.0 vdd)] ; (* vdd 0.95) ; 3.2
            "i_out_min"   [-15.0 -2.0] ; -5e-6
            "i_out_max"   [-6.0 -3.0] ; 5e-6
            ;"overshoot_r" [0.0 130.0] ; 2.0
            ;"overshoot_f" [0.0 130.0] ; 2.0
            ;"cof"         [5.0 9.0] ;  48000000.0
            "voff_stat"   [-5.0 0.0] ; 0.0025
            "voff_sys"    [-5.0 0.0] ; -0.0025
            "A"           [-10.0 -5.0] ; 7.5e-10
            #_/ }]
           [True (raise (NotImplementedError errno.ENOSYS
                                          (os.strerror errno.ENOSYS) 
                                          (.format "There is no target scaling for {} in {}"
                                                   ace-id ace-backend)))])))

(defn target-specification [^str ace-id ^dict dc ^(of list str) target-filter
                  &optional ^bool [random False] 
                            ^bool [noisy True]]
  (let [vdd (get dc (if (.startswith ace-id "op") "vsup" "vdd") "init")
        ts (cond [(in ace-id ["op1" "op6"])
                  {"a_0"         100.0
                   "ugbw"        2500000.0
                   "pm"          80.0
                   "gm"          -45.0
                   "sr_r"        2500000.0
                   "sr_f"        -2500000.0
                   "vn_1Hz"      6.0e-06
                   "vn_10Hz"     2.0e-06
                   "vn_100Hz"    6.0e-07
                   "vn_1kHz"     1.5e-07
                   "vn_10kHz"    5.0e-08
                   "vn_100kHz"   2.6e-08
                   "psrr_n"      80.0
                   "psrr_p"      80.0
                   "cmrr"        80.0
                   "v_il"        (* vdd 0.25) ; 0.9
                   "v_ih"        (* vdd 0.82) ; 2.7
                   "v_ol"        (* vdd 0.03) ; 0.1
                   "v_oh"        (* vdd 0.82) ; 2.7
                   "i_out_min"   -7e-5
                   "i_out_max"   7e-5
                   ;"overshoot_r" 0.0005
                   ;"overshoot_f" 0.0005
                   ;"cof"         48000000.0
                   "voff_stat"   0.003
                   "voff_sys"    -0.003
                   "A"           5.5e-09
                   #_/ }]
                 [(in ace-id ["op2" "op3"])
                  {"a_0"         55.0
                   "ugbw"        3500000.0
                   "pm"          65.0
                   "gm"          -20.0
                   "sr_r"        3500000.0
                   "sr_f"        -3500000.0
                   "vn_1Hz"      5e-06
                   "vn_10Hz"     2e-06
                   "vn_100Hz"    5e-07
                   "vn_1kHz"     1.5e-07
                   "vn_10kHz"    5e-08
                   "vn_100kHz"   2.5e-08
                   "cmrr"        80.0
                   "psrr_n"      60.0
                   "psrr_p"      80.0
                   "v_il"        (* vdd 0.25) ; 0.9
                   "v_ih"        (* vdd 0.95) ; 3.2
                   "v_ol"        (* vdd 0.50) ; 1.65
                   "v_oh"        (* vdd 0.95) ; 3.2
                   "i_out_min"   -7e-5
                   "i_out_max"   7e-5
                   ;"overshoot_r" 2.0
                   ;"overshoot_f" 2.0
                   ;"cof"         48000000.0
                   "voff_stat"   0.003
                   "voff_sys"    -0.003
                   "A"           5.5e-10
                   #_/ }]
                 [(in ace-id ["op4" "op5"])
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
                   "v_il"        (* vdd 0.25) ; 0.9
                   "v_ih"        (* vdd 0.95) ; 3.2
                   "v_ol"        (* vdd 0.50) ; 1.65
                   "v_oh"        (* vdd 0.95) ; 3.2
                   "i_out_min"   -7e-5
                   "i_out_max"   7e-5
                   ;"overshoot_r" 2.0
                   ;"overshoot_f" 2.0
                   "voff_stat"   3e-3
                   "voff_sys"    -1.5e-3
                   "A"           5.5e-10
                   #_/ }]
                 [(= ace-id "op8")
                  {"a_0"         65.0
                   "ugbw"        610000.0
                   "pm"          90.0
                   "gm"          -66
                   "sr_r"        500000.0
                   "sr_f"        -500000.0
                   "vn_1Hz"      1.5e-05
                   "vn_10Hz"     5.0e-06
                   "vn_100Hz"    1.5e-06
                   "vn_1kHz"     5.0e-07
                   "vn_10kHz"    1.5e-07
                   "vn_100kHz"   1.0e-07
                   "psrr_p"      130.0
                   "psrr_n"      100.0
                   "cmrr"        170.0
                   "v_il"        0.75
                   "v_ih"        3.2
                   "v_ol"        1.75
                   "v_oh"        1.75
                   "i_out_min"   -6.0e-06
                   "i_out_max"   8.5e-06
                   ;"overshoot_r" 0.00025
                   ;"overshoot_f" 0.00025
                   "voff_stat"   0.0070
                   "voff_sys"    -5.e-07
                   "A"           10.0e-10
                   #_/ }]
                 [(= ace-id "op9")
                  {"a_0"         70.0
                   "ugbw"        950000.0
                   "pm"          90.0
                   "gm"          -80.0
                   "sr_r"        900000.0
                   "sr_f"        -900000.0
                   "vn_1Hz"      5.0e-06
                   "vn_10Hz"     1.5e-06
                   "vn_100Hz"    4.5e-07
                   "vn_1kHz"     1.5e-07
                   "vn_10kHz"    7.0e-08
                   "vn_100kHz"   6.0e-08
                   "psrr_p"      130.0
                   "psrr_n"      130.0
                   "cmrr"        125.0
                   "v_il"        0.75
                   "v_ih"        5.0
                   "v_ol"        1.75
                   "v_oh"        1.75
                   "i_out_min"   -6.0e-06
                   "i_out_max"   1.0e-05
                   ;"overshoot_r" 0.0003
                   ;"overshoot_f" 0.0004
                   "voff_stat"   0.0025
                   "voff_sys"    -3.0e-05
                   "A"           8.0e-10
                   #_/ }]
                 [(= ace-id "nand4")
                  {"vs0" (/ vdd 2.0)
                   "vs1" (/ vdd 2.0)
                   "vs2" (/ vdd 2.0)
                   "vs3" (/ vdd 2.0)
                   #_/ }]
                 [(= ace-id "st1") 
                  {"v_il"  (- (* vdd 0.5) (* vdd 0.1))
                   "v_ih"  (+ (* vdd 0.5) (* vdd 0.1))
                   "t_plh" 0.8e-9
                   "t_phl" 0.8e-9
                   #_/ }]
                 [True (raise (NotImplementedError errno.ENOSYS
                            (os.strerror errno.ENOSYS) 
                            (.format "There is no target for {} in {}"
                                     ace-id ace-backend))) ])
        factor (cond [random (np.abs (np.random.normal 1 0.1))]
                     [noisy  (np.random.normal 1 0.01)]
                     [True   1.0])]
    (dfor (, p v) (.items ts)
          :if (or (in p target-filter) (empty? target-filter))
          [ p (if noisy (* v factor) v) ])))

(defn target-predicate ^(of dict str bool) [^str ace-id]
  """
  Returns True for t < p == 0 and False for t > p == 0.
  """
  (cond [(.startswith ace-id "op")
         {"A"           False
          "a_0"         True 
          "cmrr"        True 
          "cof"         True 
          "gm"          True 
          "i_out_max"   True 
          "i_out_min"   False
          "overshoot_f" False
          "overshoot_r" False
          "pm"          True 
          "psrr_n"      True 
          "psrr_p"      True 
          "sr_f"        False
          "sr_r"        True 
          "ugbw"        True 
          "v_ih"        True 
          "v_il"        False
          "v_oh"        True 
          "v_ol"        False
          "vn_1hz"      True 
          "vn_10hz"     False
          "vn_100hz"    False
          "vn_1khz"     False
          "vn_10khz"    False
          "vn_100khz"   False
          "voff_stat"   False
          "voff_sys"    True } ]
        [True
         (raise (NotImplementedError errno.ENOSYS
                     (os.strerror errno.ENOSYS) 
                     (.format "There is no target predicate for {}."
                              ace-id)))]))

(defn reward-condition ^dict [^str ace-id &optional ^float [tolerance 1e-3]]
  """
  For ace-id ≠ 'op' performance must be met within `tolerance` intervall.
  Otherwise it is a function (λ :: Performance -> Target -> Bool) that returns
  true if the performance is 'better' than target and false otherwise.
  Depending on the parameter, 'better' can mean higher or lower.
  """
  (setv == (fn [a b] (<= (/ (np.abs (- a b)) b) tolerance)))
  (cond [(.startswith ace-id "op")
         { "a_0"         <=
           "ugbw"        <=
           "pm"          <=
           ;;"gm"          >=
           "gm"          <=
           "sr_r"        <=
           "sr_f"        <=
           ;;"sr_f"        >=
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
           "cof"         <=
           "voff_stat"   >=
           ;;"voff_sys"    <=
           "voff_sys"    >=
           "A"           >=
           #_/ }]
        [(.startswith ace-id "nand")
         { "vs0" ==
           "vs1" ==
           "vs2" ==
           "vs3" ==
           #_/ }]
        [(.startswith ace-id "st")
         { "v_il"  ==
           "v_ih"  ==
           "t_plh" ==
           "t_phl" ==
           #_/ }]
        [True (raise (NotImplementedError errno.ENOSYS
                     (os.strerror errno.ENOSYS) 
                     (.format "There is no reward condition for {}."
                              ace-id)))]))
