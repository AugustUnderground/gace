(import [torch :as pt])
(import [numpy :as np])
(import [pandas :as pd])

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(require [hy.contrib.sequences [defseq seq]])
(import [hy.contrib.sequences [Sequence end-sequence]])

(defclass PrimitiveDevice []
  """
  This class represents a generic primitive device.
  """
  (defn __init__ [self]
    """
    Constructs a primitive device.
    Arguments:
      prefix:   Directory containing a torchscript *.pt and *.X *.Y scalers.
      params-x: Names of input parameters.
      params-y: Names of output parameters.
    """
    (setv self.model None))

  (defn predict [self X]
  """
  Make a prediction based on electrical characteristics.
  Arguments:
    X: pandas Data Frame containing at least `params-x`.
  """
    (with [_ (pt.no-grad)] X)))

(defclass PTM [PrimitiveDevice]
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


(defclass XFAB [PrimitiveDevice]
  """
  This class represents a specific XFAB device. It expects a scaled 
  numpy array ∈ [0;1] as input.
  """
  (defn __init__ [self ^str model-path ^str scale-y-path ^str scale-x-path]
    """
    Constructs an X-FAB XH035 primitive device trained in a specific way.
    This is not portable and only works with the 'correct' models.
    Arguments:
      model-path: Directory pointing to a torchscript *.pt model.
    """
    (setv self.path     model-path
          self.params-x ["gmoverid" "fug" "Vds" "Vbs"]
          self.params-y ["idoverw" "L" "gdsoverw" "Vgs" "vdsat" "gmbsoverw"]
          self.trafo-x  ["fug"]
          self.trafo-y  ["idoverw" "gdsoverw" "gmbsoverw"]
          self.mask-x   (lfor px self.params-x (int (in px self.trafo-x)))
          self.mask-y   (lfor py self.params-y (int (in py self.trafo-y))))

    (setv self.scale-x (jl.load scale-x-path)
          self.scale-y (jl.load scale-y-path))

    (setv self.model (-> self.path (pt.jit.load) (.cpu) (.eval))))
  
  (defn transform ^np.array [self ^np.array X ^np.array m]
    """
    Transform an array based on a given mask.
      log10(X · m) + (X * !m)
    """
    (-> X (* m) (np.ma.log10) (.filled 0) (+ (* X (- 1 m)))))

  (defn predict ^np.array [self ^np.array X &optional ^bool [scaled True]]
  """
  Make a prediction based on electrical characteristics.
  Arguments:
    X:      numpy array of inputs.
    scaled: if True (default) the input is assumed to be already transformed
            and scaled, i.e. X ∈ [0;1].
  Returns:  numpy array with outputs of machine learning model.
  """
    (-> (np.apply-along-axis 
          (fn [x]
            (with [_ (pt.no-grad)]
              (-> x (pt.from-numpy) (self.model) (.numpy))))
          1 (np.float32 (if scaled X (-> X (self.transform self.mask-x) 
                                           (self.scale-x.transform)))))
        (self.scale-y.inverse-transform)
        (self.transform self.mask-y))))
