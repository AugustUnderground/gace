(import [itertools [product]])
(import [collections.abc [Iterable]])
(import [fractions [Fraction]])
(import [decimal [Decimal]])

(import [numpy :as np])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defn map2dict [java-map]
  """
  Convert a java Map to a python dict. 
  **DOESN'T WORK FOR NESTED MAPS!**
  Arguments:
    java-map: java Map
  Returns:    dict
  """
  (dfor k (-> java-map (.keySet) (.toArray))
    [k (if (isinstance (setx w (.get java-map k)) Iterable)
        (np.array (list w)) w)]))

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

(defn dec2frac ^tuple [^float ratio]
  """
  Turns a float decimal (rounded to nearest .5) into an integer fraction.
  """
  (as-> ratio it (* it 2) (round it) (/ it 2) (str it) 
                 (Decimal it) (Fraction it) 
                 (, it.numerator it.denominator)))

(defn frac2dec ^float [^int num ^int den]
  """
  Turns a fraction into a float ratio.
  """
  (/ num den))

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
      (* (/ 100 (len A)) (np.sum (np.abs (/ (- A F) A))))))

  #@(staticmethod
    (defn SMAPE ^float [^np.array A ^np.array F]
      """
      Symmetric Mean Absolute Percentage Error:
                              n        | F_t - A_t |
        SMAPE = ( 100 / n ) · ∑    --------------------
                            t = 1   (|A_t| + |F_t|) / 2
      """ 
      (* (/ 100 (len A)) 
         (np.sum (/ (np.abs (- F A)) 
                    (/ (+ (np.abs A) (np.abs F)) 2))))))

  #@(staticmethod
    (defn MAE ^float [^np.array X ^np.array Y]
      """
      Mean Absolute Error:

               ∑ | y - x |
        MAE = ------------
                   n
      """
      (/ (np.sum (np.abs (- Y X))) (len X))))

  #@(staticmethod
    (defn MSE ^float [ ^np.array X ^np.array Y]
      """
      Mean Squared Error:
                          2
               ∑ ( y - x )
        MAE = ------------
                   n
      """
      (/ (np.sum (np.power (- X Y) 2)) (len X))))

  #@(staticmethod
    (defn RMSE ^float [^np.array X ^np.array Y]
      """
      Root Mean Squared Error:

        RMSE = √( MSE(x, y) )
      """
      (np.sqrt (Loss.MSE X Y)))))
