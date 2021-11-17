### WORK IN PROGRESS (OP8)

![op8](https://raw.githubusercontent.com/matthschw/ace/main/figures/op8.png)

_Will be_ registered as `gace:op8-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²⁷⁷ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (277 , )
              , dtype = np.float32
              , )
```

#### Action Space


| Variant | Dimensions           | Parameters                                                                                                                                                                       |
|---------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁶ ∈ [-1.0; 1.0]` | `["gmid-d1", "gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-cm4", "gmid-cm5", "fug-d1", "fug-cm1", "fug-cm2", "fug-cm3", "fug-cm4", "fug-cm5", "i1", "i2", "i3", "i4" ]`              |
| `v1`    | `ℝ ²¹ ∈ [-1.0; 1.0]` | `[ "Ld1", "Lcm1", "Lcm2", "Lcm3", "Lcm4", "Lcm5", "Wd1", "Wcm1", "Wcm2", "Wcm3", "Wcm4", "Wcm5", "Mcm1", "Mcm2", "Mcm3", "Mcm41", "Mcm51", "Mcm42", "Mcm52", "Mcm43", "Mcm53" ]` |

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
```

