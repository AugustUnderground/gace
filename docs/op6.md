### Miller Amplifier without R and C (OP6)

![op6](https://raw.githubusercontent.com/matthschw/ace/main/figures/op6.png)

Registered as `gace:op6-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²³⁵∈(-∞ ;∞)` |
| `xh018`    | `ℝ²³⁵∈(-∞ ;∞)` |
| `xt018`    | `ℝ²³⁵∈(-∞ ;∞)` |
| `sky130`   | `ℝ²³⁵∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ³⁰⁵∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (235 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (253 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (305 , )
              , dtype = np.float32
              , )
```

#### Action Space


| Variant | Dimensions       | Parameters                                                                                                                                     |
|---------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹⁴∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cs", "gmid-d", "gmid-r1", "gmid-c1", "fug-cm1", "fug-cm2", "fug-cs", "fug-d", "fug-r1", "fug-c1", "i1", "i2"]` |
| `v1`    | `ℝ¹⁸∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcs", "Lc1", "Lr1", "Wd", "Wcm1", "Wcm2", "Wcs", "Wc1", "Wr1", "Mcm11", "Mcs", "Mc1", "Mr1", "Mcm12", "Mcm13"]`       |
| `v2`    | `ℝ¹⁴∈[0,1,2]`    | `["gmid-cm1", "gmid-cm2", "gmid-cs", "gmid-d", "gmid-r1", "gmid-c1", "fug-cm1", "fug-cm2", "fug-cs", "fug-d", "fug-r1", "fug-c1", "i1", "i2"]` |
| `v3`    | `ℝ¹⁸∈[0,1,2]`    | `["Ld", "Lcm1", "Lcm2", "Lcs", "Lc1", "Lr1", "Wd", "Wcm1", "Wcm2", "Wcs", "Wc1", "Wr1", "Mcm11", "Mcs", "Mc1", "Mr1", "Mcm12", "Mcm13"]`       |

Where `i1` is the drian current through `MNCM12` and `i2` is the drain current
through `MNCM13`.

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (14 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (18 , )
              , dtype = np.float32
              , )

# v2 action space
gym.spaces.MultiDiscrete( list(repeat(3, 14))
                        , dtype = np.int32
                        , )

# v3 action space
gym.spaces.MultiDiscrete( list(repeat(3, 18))
                        , dtype = np.int32
                        , )
```

