### Un-Symmetrical Amplifier with Cascode (OP5)

![op5](https://raw.githubusercontent.com/matthschw/ace/main/figures/op5.png)

Registered as `gace:op5-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²⁸⁷∈(-∞ ;∞)` |
| `xh018`    | `ℝ²⁸⁷∈(-∞ ;∞)` |
| `xt018`    | `ℝ²⁸⁷∈(-∞ ;∞)` |
| `sky130`   | `ℝ³⁰⁵∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ³⁹⁷∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (287 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (305 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (397 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions       | Parameters                                                                                                                                                                          |
|---------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹⁶∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "gmid-c1", "gmid-r", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "fug-c1", "fug-r", "i1", "i2", "i3", "i4"]`                          |
| `v1`    | `ℝ²³∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Lc1", "Lr", "Wd", "Wcm1", "Wcm2", "Wcm3", "Wc1", "Wr", "Mcm11", "Mcm212", "Mcm31", "Mc11", "Mcm12", "Mcm222", "Mcm32", "Mc12", "Mcm13", "Mcm2x1"]` |
| `v2`    | `ℝ³³∈[0,1]`    | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "gmid-c1", "gmid-r", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "fug-c1", "fug-r", "i1", "i2", "i3", "i4"]`                          |
| `v3`    | `ℝ⁴⁷∈[0,1]`    | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Lc1", "Lr", "Wd", "Wcm1", "Wcm2", "Wcm3", "Wc1", "Wr", "Mcm11", "Mcm212", "Mcm31", "Mc11", "Mcm12", "Mcm222", "Mcm32", "Mc12", "Mcm13", "Mcm2x1"]` |


Where `i1` is the drain current through `MNCM13`, `i2` is the drain current
through `MPCM212`, `i3` is the drain current through `MPCM222` and `i4` is the
drain current through `MNCM12`.

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (16 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (23 , )
              , dtype = np.float32
              , )

# v2 action space
gym.spaces.Discrete( 33 )

# v3 action space
gym.spaces.Discrete( 47 )
```

