(import [enum [Enum]])
(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [fractions [Fraction]])
(import [decimal [Decimal]])
(import [functools [partial]])
(import [operator [itemgetter]])

(import [numpy :as np])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defclass AmplifierID [Enum] 
  """
  Supported / Available Operational Amplifieres
  """
  (setv MILLER      1
        SYMMETRICAL 2))

(defclass TechID [Enum] 
  """
  Supported / Available Technologies
  """
  (setv XH035    1
        SKYWATER 2
        PTM      3))

(defn initial-sizing ^dict [^AmplifierID amp ^TechID pdk]
  """
  For a given Operational Amplifier and Technology, return an initial sizing.
  """
  (cond [(and (= amp AmplifierID.SYMMETRICAL) (= pdk TechID.XH035))
         {"Wcm1"  2.55e-6
          "Wcm2"  10.65e-6
          "Wcm3"  6.5e-6
          "Wd"    3e-6
          "Lcm1"  1.95e-6
          "Lcm2"  3.35e-6
          "Lcm3"  2.6e-6
          "Ld"    5.15e-6
          "Mcm11" 1.0
          "Mcm12" 2.0
          "Mcm21" 1.0
          "Mcm22" 4.0
          "Mcm31" 2.0
          "Mcm32" 2.0
          "Md"    2.0
          #_/     }]
        [(and (= amp AmplifierID.MILLER) (= pdk TechID.XH035))
         {"Wcm1"  2.6e-6
          "Wcm2"  6.9e-6
          "Wcs"   3.35e-6
          "Wd"    4e-6
          "Lcm1"  2e-6
          "Lcm2"  1.7e-6
          "Lcs"   1e-6
          "Ld"    2.5e-6
          "Lres"  131e-6
          "Wres"  2e-6
          "Wcap"  69e-6
          "Mcm11" 1.0
          "Mcm12" 5.0
          "Mcm13" 22.0
          "Mcs"   22.0
          "Md"    2.0
          "Mcm21" 2.0
          "Mcm22" 2.0
          "Mcap"  1.0
          #_/     }]
        [True
         (raise (NotImplementedError 
                 f"Given {amp.name} or {pdk.name} are not implemented"))]))

(defn random-sizing [^AmplifierID amp ^dict tech-cfg]
  (cond [(= amp AmplifierID.MILLER) 
         {"Wcm1" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                              (get tech-cfg "mos" "W" "max")
                                              (get tech-cfg "grid")))
          "Wcm2" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                              (get tech-cfg "mos" "W" "max")
                                              (get tech-cfg "grid")))
          "Wcs" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                             (get tech-cfg "mos" "W" "max")
                                             (get tech-cfg "grid")))
          "Wd" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                            (get tech-cfg "mos" "W" "max")
                                            (get tech-cfg "grid")))
          "Lcm1" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                              (get tech-cfg "mos" "L" "max")
                                              (get tech-cfg "grid")))
          "Lcm2" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                              (get tech-cfg "mos" "L" "max")
                                              (get tech-cfg "grid")))
          "Lcs" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                             (get tech-cfg "mos" "L" "max")
                                             (get tech-cfg "grid")))
          "Ld" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                            (get tech-cfg "mos" "L" "max")
                                            (get tech-cfg "grid")))
          "Wres" (np.random.choice (np.arange (get tech-cfg "res" "W" "min")
                                              (get tech-cfg "res" "W" "max")
                                              (get tech-cfg "grid")))
          "Lres" (np.random.choice (np.arange (get tech-cfg "res" "L" "min")
                                              (get tech-cfg "res" "L" "max")
                                              (get tech-cfg "grid")))
          "Wcap" (np.random.choice (np.arange (get tech-cfg "cap" "W" "min")
                                              (get tech-cfg "cap" "W" "max")
                                              (get tech-cfg "grid")))
          "Mcm11" (np.random.randint 1 3)
          "Mcm12" (np.random.randint 1 16)
          "Mcm13" (np.random.randint 1 51)
          "Mcs" (np.random.randint 1 51)
          "Md" 2.0
          "Mcm21" 2.0
          "Mcm22" 2.0
          "Mcap" 1.0 }]
        [(= amp AmplifierID.SYMMETRICAL)
         {"Wcm1" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                              (get tech-cfg "mos" "W" "max")
                                              (get tech-cfg "grid")))
          "Wcm2" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                              (get tech-cfg "mos" "W" "max")
                                              (get tech-cfg "grid")))
          "Wcm3" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                              (get tech-cfg "mos" "W" "max")
                                              (get tech-cfg "grid")))
          "Wd" (np.random.choice (np.arange (get tech-cfg "mos" "W" "min")
                                            (get tech-cfg "mos" "W" "max")
                                            (get tech-cfg "grid")))
          "Lcm1" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                              (get tech-cfg "mos" "L" "max")
                                              (get tech-cfg "grid")))
          "Lcm2" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                              (get tech-cfg "mos" "L" "max")
                                              (get tech-cfg "grid")))
          "Lcm3" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                              (get tech-cfg "mos" "L" "max")
                                              (get tech-cfg "grid")))
          "Ld" (np.random.choice (np.arange (get tech-cfg "mos" "L" "min")
                                            (get tech-cfg "mos" "L" "max")
                                            (get tech-cfg "grid")))
          "Mcm11" (np.random.randint 1 2)
          "Mcm12" (np.random.randint 1 15)
          "Mcm21" (np.random.randint 1 50)
          "Mcm22" (np.random.randint 1 50)
          "Mcm31" (np.random.randint 1 50)
          "Mcm32" (np.random.randint 1 50)
          "Md" 2.0}]
        [True
         (raise (NotImplementedError 
                 f"Given {amp.name} is not implemented"))]))

(defn calculate-area [^AmplifierID amp ^dict sizing]
  (cond [(= amp AmplifierID.MILLER)
         (let [(, Mcm11 Mcm12 Mcm13 Wcm1 Lcm1 
                  Mdp1 Wdp1 Ldp1
                  Mcm21 Mcm22 Wcm2 Lcm2 
                  Mcs1 Wcs1 Lcs1
                  Wres Lres Mcap Wcap) 
                    ((itemgetter "Mcm11" "Mcm12" "Mcm13" "Wcm1" "Lcm1" "Md" "Wd" 
                                 "Ld" "Mcm21" "Mcm22" "Wcm2" "Lcm2" "Mcs" "Wcs"
                                 "Lcs" "Wres" "Lres" "Mcap" "Wcap")
                               sizing)]
          (+ (* 1.0 (+ Mcm11 Mcm12 Mcm13) Wcm1 Lcm1)
             (* 2.0 Mdp1 Wdp1 Ldp1)
             (* 2.0 (+ Mcm21 Mcm22) Wcm2 Lcm2)
             (* 1.0 Mcs1 Wcs1 Lcs1)
             (* 1.0 Wres Lres)
             (* 1.0 Mcap Wcap Wcap)))]
        [(= amp AmplifierID.SYMMETRICAL)
         (let [(, Mcm11 Mcm12 Wcm1 Lcm1 
                  Mdp1 Wdp1 Ldp1
                  Mcm21 Mcm22 Wcm2 Lcm2 
                  Mcm31 Mcm32 Wcm3 Lcm3) 
                    ((itemgetter "Mcm11" "Mcm12" "Wcm1" "Lcm1" 
                                 "Md" "Wd" "Ld" 
                                 "Mcm21" "Mcm22" "Wcm2" "Lcm2" 
                                 "Mcm31" "Mcm32" "Wcm3" "Lcm3")
                               sizing)]
          (+ (* 1.0 (+ Mcm11 Mcm12) Wcm1 Lcm1)
             (* 2.0 Mdp1 Wdp1 Ldp1)
             (* 2.0 (+ Mcm21 Mcm22) Wcm2 Lcm2)
             (* 1.0 (+ Mcm31 Mcm32) Wcm3 Lcm3)))]
        [True
         (raise (NotImplementedError 
                 f"Given {amp.name} is not implemented"))]))

(defn scale-value ^float [^float x ^float x-min ^float x-max
                &optional ^float [a -1.0] ^float [b 1.0]]
  """
  Scales a value s.t. x′∈ [a;b], where a = -1.0 and b = 1.0 by default.

              (x - x_min) · (b - a)
    x′ = a + -----------------------
                (x_max - x_min)
  """
  (+ a (/ (* (- x x-min) (- b a)) (- x-max x-min))))

(defn unscale-value ^float [^float x′ ^float x-min ^float x-max
                  &optional ^float [a -1.0] ^float [b 1.0]]
  """
  Scales a value x′∈ [a;b] back to its original, where a = -1.0 and b = 1.0 by
  default.

                (x′- a)
    x = x_min + ------- · (x_max - x_min)
                (b - a)
  """
  (+ x-min (* (/ (- x′ a) (- b a)) (- x-max x-min))))

(defn dec-to-frac ^tuple [^float ratio]
  """
  Turns a float decimal (rounded to nearest .5) into an integer fraction.
  """
  (as-> ratio it (* it 2) (round it) (/ it 2) (str it) 
                 (Decimal it) (Fraction it) 
                 (, it.numerator it.denominator)))

(defn frac-to-dec ^float [^int num ^int den]
  """
  Turns a fraction into a float ratio.
  """
  (/ num den))

(defn ape [t o] 
  """
  Absolute Percentage Error for scalar values.
  """
  (* 100 (/ (np.abs (- t o)) 
            (if (!= t 0) t 1))))

(defn absolute-condition [t c] 
  """
  Returns a function for reward calculation based on the given target `t` and a
  conditional predicate `c`. If the target meets the conditional the reward is
  calculated as: 
            - ape(x)
    r(x) = -e         + 1
  otherwise it is:
    r(x) = - ape(x)
  .
  """
  (let [cn (partial (eval c) t)
      er (partial ape t)]
    (fn [x] 
      (if (cn x) 
         (+ (- (np.exp (- (er x)))) 1) 
         (- (er x))))))


(defn ranged-condition [l u] 
  """
  Returns a function for reward calculation based on the given lower `l` and
  upper `u` bounds. See `absolute-condition` for details.
  """
  (let [er (partial ape (np.abs (- l u)))]
    (fn [x] 
      (if (and (<= l x) (>= u x)) 
         (+ (- (np.exp (- (er x)))) 1)
         (- (er x))))))

(defclass Loss []
  """
  A purely static collection of loss functions. All the functions here can
  only operate on numpy arrays.
  """

  #@(staticmethod
    (defn MAPE ^float [^np.array A ^np.array F]
      """
      Mean Absolute Percentage Error:
                             n
        MAPE = ( 100 / n ) · ∑  | (A_t - F_t) / A_t |
                           t = 1
      """
      (.item (* (/ 100 (len A)) 
                (np.sum (np.abs (np.divide (- A F) A :where (!= A 0))))))))

  #@(staticmethod
    (defn SMAPE ^float [^np.array A ^np.array F]
      """
      Symmetric Mean Absolute Percentage Error:
                              n        | F_t - A_t |
        SMAPE = ( 100 / n ) · ∑    --------------------
                            t = 1   (|A_t| + |F_t|) / 2
      """ 
      (.item (* (/ 100 (len A)) 
                  (np.sum (/ (np.abs (- F A)) 
                             (/ (+ (np.abs A) (np.abs F)) 2)))))))

  #@(staticmethod
    (defn MAE ^float [^np.array X ^np.array Y]
      """
      Mean Absolute Error:

               ∑ | y - x |
        MAE = -------------
                    n
      """
      (.item (/ (np.sum (np.abs (- Y X))) (len X)))))

  #@(staticmethod
    (defn MSE ^float [ ^np.array X ^np.array Y]
      """
      Mean Squared Error:
                          2
               ∑ ( y - x )
        MAE = -------------
                    n
      """
      (.item (/ (np.sum (np.power (- X Y) 2)) (len X)))))

  #@(staticmethod
    (defn RMSE ^float [^np.array X ^np.array Y]
      """
      Root Mean Squared Error:

        RMSE = √( MSE(x, y) )
      """
      (.item (np.sqrt (Loss.MSE X Y))))))








;(defclass ACL []
;  """
;  REST API Interface to analog circuit library.
;  """
;  (defn __init__ [self &optional ^str [hostname "localhost"]
;                                 ^int [port 8888]]
;    """
;    Analog Circuit Library Interface face.
;    Arguments:
;      hostname    default 'localhost'
;      port        default '8888'
;    Make sure server is actually running.
;    """
;    (setv self.base-url f"http://{hostname}:{port}"))
;
;  (defn evaluate-circuit ^dict [self ^AmplifierID amp &optional ^dict [sizing {}]]
;    """
;    Run simulation and return results.
;    Arguments:
;      amp     The amplifier (AmplifierID) to simulate.
;      sizing  Device sizes for the given circuit.
;    Returns:
;      Circuit Performance.
;    """
;    (let [url (.format "{}/sim/op{}" self.base-url amp.value)
;        params (dfor (, k v) (.items sizing)
;                  [k (if (isinstance v list) v [v])])]
;      (-> req (.post url :json params) (.json))))
;
;  (defn _sizing ^dict [self ^AmplifierID amp ^str sizing]
;    """
;    Meta function for getting sizing parameters for a given AmplifierID, where
;    sizing = 'rng' | 'init'
;    """
;    (let [url (.format "{}/{}/op{}" self.base-url sizing amp.value)]
;      (-> req (.get url) (.json))))
;
;  (defn random-sizing ^dict [self ^AmplifierID amp]
;    """
;    Get random sizing for given AmplifierID.
;    """
;    (self._sizing amp "rng"))
;
;  (defn initial-sizing ^dict [self ^AmplifierID amp]
;    """
;    Get curated / good sizing for given AmplifierID.
;    """
;    (self._sizing amp "init"))
;
;  (defn _params ^list [self ^AmplifierID amp ^str p]
;    """
;    Meta function for getting available keys for a given AmplifierID, where
;      keys = 'params' | 'perfs'
;    """
;    (let [p-route (cond [(= p "parameters") "params"]
;                   [(= p "perforamnces") "perfs"])
;        url (.format "{}/{}/op{}" self.base-url p-route amp.value)]
;      (-> req (.get url) (.json) (get p))))
;
;  (defn parameters ^list [self ^AmplifierID amp]
;    """
;    Get available sizing parameters for given AmplifierID.
;    """
;    (self._params amp "parameters"))
;
;  (defn performances ^list [self ^AmplifierID amp]
;    """
;    Get available perforamnce parameters for given AmplifierID.
;    """
;    (self._params amp "perforamnces")))
