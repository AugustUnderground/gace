(import gym)
(import warnings)
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

(defn MAPE ^float [^np.array A ^np.array F]
  """
  Mean Absolute Percentage Error:
                         n
    MAPE = ( 100 / n ) · ∑  | (A_t - F_t) / A_t |
                       t = 1
  """
  (.item (* (/ 100 (len A)) 
            (np.sum (np.abs (np.divide (- A F) A :where (!= A 0)))))))

(defn SMAPE ^float [^np.array A ^np.array F]
  """
  Symmetric Mean Absolute Percentage Error:
                          n        | F_t - A_t |
    SMAPE = ( 100 / n ) · ∑    --------------------
                        t = 1   (|A_t| + |F_t|) / 2
  """ 
  (.item (* (/ 100 (len A)) 
              (np.sum (/ (np.abs (- F A)) 
                         (/ (+ (np.abs A) (np.abs F)) 2))))))

(defn MAE ^float [^np.array X ^np.array Y]
  """
  Mean Absolute Error:

           ∑ | y - x |
    MAE = -------------
                n
  """
  (.item (/ (np.sum (np.abs (- Y X))) (len X))))

(defn MSE ^float [ ^np.array X ^np.array Y]
  """
  Mean Squared Error:
                      2
           ∑ ( y - x )
    MAE = -------------
                n
  """
  (.item (/ (np.sum (np.power (- X Y) 2)) (len X))))

(defn RMSE ^float [^np.array X ^np.array Y]
  """
  Root Mean Squared Error:

    RMSE = √( MSE(x, y) )
  """
  (.item (np.sqrt (Loss.MSE X Y))))
