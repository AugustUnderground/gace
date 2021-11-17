### Miller Amplifier (OP1)

![op1](https://github.com/matthschw/ace/blob/main/figures/op1.png)

Registered as `gace:op1-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²¹¹ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box(low   = -np.inf
              , high  = np.inf
              , shape = (211 , )
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

`ℝ ¹² ∈ [-1.0; 1.0]` 

</td>
<td>

```python
[ "gmid-cm1", "gmid-cm2", "gmid-cs", "gmid-d"
, "fug-cm1", "fug-cm2", "fug-cs", "fug-d" 
, "rc", "cc", "i1", "i2" ]
```

</td>
</tr>
<tr> 
<td> 

`v1` 

</td> 
<td> 

`ℝ ¹⁵ ∈ [-1.0; 1.0]` 

</td>
<td>

```python
[ "Ld", "Lcm1",  "Lcm2",  "Lcs",        "Lrc"
, "Wd", "Wcm1",  "Wcm2",  "Wcs", "Wcc", "Wrc"
      , "Mcm11",          "Mcs"
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


