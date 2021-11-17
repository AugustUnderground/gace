### Symmetrical Amplifier (OP2)

![op2](https://raw.githubusercontent.com/matthschw/ace/main/figures/op2.png)

Registered as `gace:op2-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²⁰⁶ ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ²⁶⁶ ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ²⁹⁴ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (206 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (266 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (294 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions       | Parameters                                                                                            |
|---------|------------------|-------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹⁰∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "i1" "i2"]` |
| `v1`    | `ℝ¹²∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Wd", "Wcm1", "Wcm2", "Wcm3", "Mcm11", "Mcm21", "Mcm12", "Mcm22"]`    |

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (10 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (12 , )
              , dtype = np.float32
              , )
```


