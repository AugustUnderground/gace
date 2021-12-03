### Amplifier with Wideswing Current Mirror (OP8)

![op8](https://raw.githubusercontent.com/matthschw/ace/main/figures/op8.png)

_Will be_ registered as `gace:op8-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²⁷⁷∈(-∞ ;∞)` |
| `xh018`    | `ℝ²⁷⁷∈(-∞ ;∞)` |
| `xt018`    | `ℝ²⁷⁷∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (277 , )
              , dtype = np.float32
              , )
```

#### Action Space


| Variant | Dimensions       | Parameters                                                                                                                                                                       |
|---------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹⁶∈[-1.0;1.0]` | `["gmid-d1", "gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-cm4", "gmid-cm5", "fug-d1", "fug-cm1", "fug-cm2", "fug-cm3", "fug-cm4", "fug-cm5", "i1", "i2", "i3", "i4" ]`              |
| `v1`    | `ℝ²¹∈[-1.0;1.0]` | `[ "Ld1", "Lcm1", "Lcm2", "Lcm3", "Lcm4", "Lcm5", "Wd1", "Wcm1", "Wcm2", "Wcm3", "Wcm4", "Wcm5", "Mcm1", "Mcm2", "Mcm3", "Mcm41", "Mcm51", "Mcm42", "Mcm52", "Mcm43", "Mcm53" ]` |
| `v2`    | `ℝ¹⁶∈[0,1,2]`    | `["gmid-d1", "gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-cm4", "gmid-cm5", "fug-d1", "fug-cm1", "fug-cm2", "fug-cm3", "fug-cm4", "fug-cm5", "i1", "i2", "i3", "i4" ]`              |
| `v3`    | `ℝ²¹∈[0,1,2]`    | `[ "Ld1", "Lcm1", "Lcm2", "Lcm3", "Lcm4", "Lcm5", "Wd1", "Wcm1", "Wcm2", "Wcm3", "Wcm4", "Wcm5", "Mcm1", "Mcm2", "Mcm3", "Mcm41", "Mcm51", "Mcm42", "Mcm52", "Mcm43", "Mcm53" ]` |

Where `i1` is the drain current through `MNCM53`, `i2` is the drain current
through `MNCM52`, `i3` is the drain current through `MPCM42` and `i4` is the
drain current through `MPCM43`.

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
              , shape = (21 , )
              , dtype = np.float32
              , )

# v2 action space
gym.spaces.MultiDiscrete( list(repeat(3, 16))
                        , dtype = np.int32
                        , )

# v3 action space
gym.spaces.MultiDiscrete( list(repeat(3, 22))
                        , dtype = np.int32
                        , )
```

