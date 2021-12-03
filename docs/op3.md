### Un-Symmetrical Amplifier (OP3)

![op3](https://raw.githubusercontent.com/matthschw/ace/main/figures/op3.png)

Registered as `gace:op3-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²⁴⁶∈(-∞ ;∞)` |
| `xh018`    | `ℝ²⁴⁶∈(-∞ ;∞)` |
| `xt018`    | `ℝ²⁴⁶∈(-∞ ;∞)` |
| `sky130`   | `ℝ²⁶⁶∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ³⁰⁶∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (246 , )
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
              , shape = (306 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions       | Parameters                                                                                                                       |
|---------|------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹¹∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "i1", "i2", "i3" ]`                    |
| `v1`    | `ℝ¹⁵∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Wd", "Wcm1", "Wcm2", "Wcm3", "Mcm11", "Mcm212", "Mcm31", "Mcm12", "Mcm222", "Mcm32", "Mcm2x1"]` |
| `v2`    | `ℝ¹¹∈[0,1,2]`    | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "i1", "i2", "i3" ]`                    |
| `v3`    | `ℝ¹⁵∈[0,1,2]`    | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Wd", "Wcm1", "Wcm2", "Wcm3", "Mcm11", "Mcm212", "Mcm31", "Mcm12", "Mcm222", "Mcm32", "Mcm2x1"]` |

Where `i1` is the drain current through `MNCM12`, `i2` is the drain current
through `MPCM212` and `i3` is the drain current through `MPCM222`.

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (11 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (15 , )
              , dtype = np.float32
              , )

# v2 action space
gym.spaces.MultiDiscrete( list(repeat(3, 11))
                        , dtype = np.int32
                        , )

# v3 action space
gym.spaces.MultiDiscrete( list(repeat(3, 15))
                        , dtype = np.int32
                        , )
```

