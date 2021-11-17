### Miller Amplifier without R and C (OP6)

![op6](https://raw.githubusercontent.com/matthschw/ace/main/figures/op6.png)

Registered as `gace:op6-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²³⁵ ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ²³⁵ ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ²⁷⁵ ∈ (-∞ ; ∞)` |

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
              , shape = (235 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (275 , )
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

`ℝ ¹⁴ ∈ [-1.0; 1.0]`

</td>
<td>

```python
[ "gmid-cm1", "gmid-cm2", "gmid-cs", "gmid-d", "gmid-r1", "gmid-c1"
, "fug-cm1",  "fug-cm2",  "fug-cs",  "fug-d",  "fug-r1",  "fug-c1"
, "i1", "i2" ]
```

</td>
</tr>
<tr> 
<td> 

`v1` 

</td> 
<td> 

`ℝ ¹⁸ ∈ [-1.0; 1.0]`

</td>
<td>

```python
[ "Ld", "Lcm1",  "Lcm2",  "Lcs", "Lc1", "Lr1"
, "Wd", "Wcm1",  "Wcm2",  "Wcs", "Wc1", "Wr1"
      , "Mcm11",          "Mcs", "Mc1", "Mr1" 
      , "Mcm12"
      , "Mcm13" ]
```

</td>
</tr>
</table>

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
```


