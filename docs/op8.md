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
[ "gmid-d1", "gmid-cm1", "gmid-cm2", "gmid-cm3", "gmid-cm4", "gmid-cm5" 
, "fug-d1",  "fug-cm1",  "fug-cm2",  "fug-cm3",  "fug-cm4",  "fug-cm5" 
, "i1", "i2", "i3", "i4" ]
```

</td>
</tr>
<tr> 
<td> 

`v1` 

</td> 
<td> 

`ℝ ²¹ ∈ [-1.0; 1.0]`

</td>
<td>

```python
[ "Ld1", "Lcm1", "Lcm2", "Lcm3", "Lcm4",  "Lcm5"
, "Wd1", "Wcm1", "Wcm2", "Wcm3", "Wcm4",  "Wcm5"
       , "Mcm1", "Mcm2", "Mcm3", "Mcm41", "Mcm51" 
                               , "Mcm42", "Mcm52"
                               , "Mcm43", "Mcm53" ]
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
              , shape = (21 , )
              , dtype = np.float32
              , )
```


