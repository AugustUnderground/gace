### Un-Symmetrical Amplifier with Cascode (OP5)

![op5](https://github.com/matthschw/ace/blob/main/figures/op5.png)

Registered as `gace:op5-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²⁸⁵ ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ³⁰⁵ ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ³⁸³ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (285 , )
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
              , shape = (383 , )
              , dtype = np.float32
              , )
```

#### Action Space

<table>
<tr><th>Variant</th><th>Dimensions</th> <th>Parameters</th></tr>
<tr> 
<td> 

`v0` 

</td> 
<td> 

`ℝ ¹⁶ ∈ [-1.0; 1.0]`

</td>
<td>

```python
[ "gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-d", "gmid-c1", "gmid-r"
, "fug-cm1",  "fug-cm2",  "fug-cm3",  "fug-d",  "fug-c1",  "fug-r"
, "i1", "i2", "i3", "i4" ]
```

</td>
</tr>
<tr> 
<td> 

`v1` 

</td> 
<td> 

`ℝ ²² ∈ [-1.0; 1.0]`

</td>
<td>

```python
[ "Ld", "Lcm1",  "Lcm2",   "Lcm3",  "Lc1",  "Lr"
, "Wd", "Wcm1",  "Wcm2",   "Wcm3",  "Wc1",  "Wr"
      , "Mcm11", "Mcm212", "Mcm31", "Mc11" 
      , "Mcm12", "Mcm222", "Mcm32", "Mc12" 
      , "Mcm13", "Mcm2x1"                          ]
```

</td>
</tr>
</table>

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
              , shape = (22 , )
              , dtype = np.float32
              , )
```


