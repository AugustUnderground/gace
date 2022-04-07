### Symmetrical Amplifier (OP2)

![op2](https://raw.githubusercontent.com/matthschw/ace/main/figures/op2.png)

Registered as `gace:op2-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²⁴⁸∈(-∞ ;∞)` |
| `xh018`    | `ℝ²⁴⁸∈(-∞ ;∞)` |
| `xt018`    | `ℝ²⁴⁸∈(-∞ ;∞)` |
| `sky130`   | `ℝ²⁷⁴∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ³¹⁴∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (248 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (274 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (314 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions       | Parameters                                                                                            |
|---------|------------------|-------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹⁰∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "i1" "i2"]` |
| `v1`    | `ℝ¹⁵∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Wd", "Wcm1", "Wcm2", "Wcm3", "Mcm11", "Mcm21", "Mcm12", "Mcm22"]`    |
| `v2`    | `ℝ²¹∈[0,1]`      | `["gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "fug-cm1", "fug-cm2", "fug-cm3", "fug-d", "i1" "i2"]` |
| `v3`    | `ℝ³¹∈[0,1]`      | `["Ld", "Lcm1", "Lcm2", "Lcm3", "Wd", "Wcm1", "Wcm2", "Wcm3", "Mcm11", "Mcm21", "Mcm12", "Mcm22"]`    |

Where `i1` is the drain current through `MNCM12` and `i2` is the drain current
through `MPCM212` and `MPCM222`.

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
              , shape = (15 , )
              , dtype = np.float32
              , )

# v2 action space
gym.spaces.Discrete( 21 )

# v3 action space
gym.spaces.Discrete( 31 )
```

