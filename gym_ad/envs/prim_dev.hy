(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])
(import [joblib :as jl])

(import [precept [PreceptModule PreceptDataFrameModule]])

(import [.util [*]])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defclass PrimitiveDevice []
  """
  This class represents a specific XFAB device. It expects a scaled 
  numpy array ∈ [0;1] as input.
  """
  (defn __init__ [self ^str model-path ^str scale-x-path ^str scale-y-path]
    """
    Constructs a primitive device that can handle pre-scaled input for
    predictions: X ∈ [0;1].
    This is not portable and only works with the 'correct' models.
    Arguments:
      model-path: Directory pointing to a torchscript *.pt model.
    """
    (setv self.path     model-path
          self.params-x ["gmoverid" "fug" "Vds" "Vbs"]
          self.params-y ["idoverw" "L" "gdsoverw" "Vgs"]
          self.trafo-x  ["fug"]
          self.trafo-y  ["idoverw" "gdsoverw"]
          self.mask-x   (np.array (lfor px self.params-x (int (in px self.trafo-x))))
          self.mask-y   (np.array (lfor py self.params-y (int (in py self.trafo-y)))))

    (setv self.scaler-x (jl.load scale-x-path)
          self.scaler-y (jl.load scale-y-path)
          self.scale-x  (fn [X] (self.scaler-x.transform X))
          self.scale-y  (fn [Y] (self.scaler-y.inverse-transform Y)))

    (setv self.trafo-x  (fn [X] (+ (* (np.log10 (np.abs X) 
                                                :where (> (np.abs X) 0)) 
                                      self.mask-x) 
                                   (* X (- 1 self.mask-x))))
          self.trafo-y  (fn [Y] (+ (* (np.power 10 Y) self.mask-y) 
                                   (* Y (- 1 self.mask-y)))))

    (setv self.model (-> self.path 
                         (PreceptModule.load-from-checkpoint)
                         (.cpu) (.eval))))
  
  (defn predict ^np.array [self ^np.array X]
  """
  Make a prediction based on electrical characteristics.
  Arguments:
    X:      numpy array of inputs. Shape: (1, 4)
  Returns:  numpy array with re-scaled and transformed outputs of machine
            learning model.
  """
    (with [_ (pt.no-grad)]
      (-> X (self.trafo-x) 
            (self.scale-x) 
            (np.float32) 
            (pt.from-numpy) 
            (self.model) 
            (.numpy) 
            (self.scale-y) 
            (self.trafo-y)))))

(defclass PrimitiveDeviceDf []
  """
  This class represents a predicitve technology model primitive device, with
  custom scaling.
  """
  (defn __init__ [self ^str prefix ^list params-x ^list params-y]
    """
    Constructs a primitive device.
    Arguments:
      prefix:   Directory containing a torchscript *.pt and *.X *.Y scalers.
      params-x: Names of input parameters.
      params-y: Names of output parameters.
    """
    (setv self.prefix prefix
          self.params-x params-x
          self.params-y params-y)

    (setv self.model (pt.jit.load f"{self.prefix}.pt")
          self.scale-x (jl.load f"{self.prefix}.X")
          self.scale-y (jl.load f"{self.prefix}.Y"))
    
    (-> self.model (.cpu) (.eval)))

  (defn predict ^pd.DataFrame [self ^pd.DataFrame X]
    """
    Make a prediction based on electrical characteristics.
    Arguments:
      X: pandas Data Frame containing at least `params-x`.
    """
    (with [_ (pt.no-grad)]
      (let [_ (setv X.fug (np.log10 X.fug.values))
            X′ (-> X (get self.params-x) (. values) (self.scale-x.transform))
            Y′ (-> X′ (np.float32) (pt.from-numpy) (self.model) (.numpy))
            Y  (pd.DataFrame (self.scale-y.inverse-transform Y′)
                             :columns self.params-y)]
        (setv Y.jd (np.power 10 Y.jd.values))
        (setv Y.gdsw (np.power 10 Y.gdsw.values))
        Y))))

(defclass PrimitiveDeviceTs []
  """
  This class represents a specific XFAB device. It expects a scaled 
  numpy array ∈ [0;1] as input.
  """
  (defn __init__ [self ^str model-path ^str scale-x-path ^str scale-y-path]
    """
    Constructs a primitive device that can handle pre-scaled input for
    predictions: X ∈ [0;1].
    This is not portable and only works with the 'correct' models.
    Arguments:
      model-path: Directory pointing to a torchscript *.pt model.
    """
    (setv self.path     model-path
          self.params-x ["gmoverid" "fug" "Vds" "Vbs"]
          self.params-y ["idoverw" "L" "gdsoverw" "Vgs"]
          self.trafo-x  ["fug"]
          self.trafo-y  ["idoverw" "gdsoverw"]
          self.mask-x   (np.array (lfor px self.params-x (int (in px self.trafo-x))))
          self.mask-y   (np.array (lfor py self.params-y (int (in py self.trafo-y)))))

    (setv self.scaler-x (jl.load scale-x-path)
          self.scaler-y (jl.load scale-y-path)
          self.scale-x  (fn [X] (self.scaler-x.transform X))
          self.scale-y  (fn [Y] (self.scaler-y.inverse-transform Y)))

    (setv self.trafo-x  (fn [X] (+ (* (np.log10 (np.abs X) 
                                                :where (> (np.abs X) 0)) 
                                      self.mask-x) 
                                   (* X (- 1 self.mask-x))))
          self.trafo-y  (fn [Y] (+ (* (np.power 10 Y) self.mask-y) 
                                   (* Y (- 1 self.mask-y)))))

    (setv self.model (-> self.path (pt.jit.load) (.cpu) (.eval))))
  
  (defn predict ^np.array [self ^np.array X]
  """
  Make a prediction based on electrical characteristics.
  Arguments:
    X:      numpy array of inputs. Shape: (1, 4)
  Returns:  numpy array with re-scaled and transformed outputs of machine
            learning model.
  """
    (with [_ (pt.no-grad)]
      (-> X (self.trafo-x) 
            (self.scale-x) 
            (np.float32) 
            (pt.from-numpy) 
            (self.model) 
            (.numpy) 
            (self.scale-y) 
            (self.trafo-y)))))
