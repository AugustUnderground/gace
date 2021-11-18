### Miller Amplifier (OP1)

![op1](https://raw.githubusercontent.com/matthschw/ace/main/figures/op1.png)


Registered as `gace:op1-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions     |
|------------|----------------|
| `xh035`    | `ℝ²¹¹∈(-∞ ;∞)` |
| `sky130`   | `ℝ²²⁹∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ²⁶¹∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035
gym.spaces.Box(low   = -np.inf
              , high  = np.inf
              , shape = (211 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box(low   = -np.inf
              , high  = np.inf
              , shape = (229 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box(low   = -np.inf
              , high  = np.inf
              , shape = (261 , )
              , dtype = np.float32
              , )
```

#### Action Space 

| Variant | Dimensions       | Parameters                                                                                                          |
|---------|------------------|---------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ¹²∈[-1.0;1.0]` | `["gmid-cm1", "gmid-cm2", "gmid-cs", "gmid-d", "fug-cm1", "fug-cm2", "fug-cs", "fug-d" , "rc", "cc", "i1", "i2"]`   |
| `v1`    | `ℝ¹⁵∈[-1.0;1.0]` | `["Ld", "Lcm1", "Lcm2", "Lcs", "Lrc", "Wd", "Wcm1", "Wcm2", "Wcs", "Wcc", "Wrc", "Mcm11", "Mcs", "Mcm12", "Mcm13"]` |

Where `i1` is the drain current through `MNCM1A` and `i2` is the drain current
through `MNCM1B`.

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (12 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (14 , )
              , dtype = np.float32
              , )

```

