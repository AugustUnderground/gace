## Inverter Environments

### Observation Spaces and Technologies

The observation space includes the current perforamnce, the specified target,
the normalized distance between target and current performance (`| t - p | / t`) 
and operating point characteristics for all devices in the circuit.

```json
{ "performance": { "vs0": "Switching Voltage 0"
                 , "vs2": "Switching Voltage 1"
                 , "vs1": "Switching Voltage 2"
                 , "vs3": "Switching Voltage 3" }
, "target":      { "Same keys as 'performance'": "..." }
, "distance":    { "Same keys as 'performance'": "..." }
, "state":       { "electrical characteristics": "..." } }
```

Since there are only comparatively few observations, they are identical across
technologies.

### Action Spaces and Variants

Action Spaces in `v1` are _continuous_ and implemented with
`gym.spaces.Box`. For further details, see the descriptions for specific
environments. For now, only `v1` is implemented for Inverter Environments.

### Reward

The overall reward `r` is calculated based on individual performance
parameters:

```
r = - ∑ ( | t - p | / t )
```

### 4 Gate Inverter Chain (NAND4)

![nand4](https://github.com/matthschw/ace/blob/main/figures/nand4.png)

Registered as `gace:nand4-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions        |
|------------|-------------------|
| `xh035`    | `ℝ ¹² ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ¹² ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ¹² ∈ (-∞ ; ∞)` |

```python
# xh035, sky130, gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (12 , )
              , dtype = np.float32
              , )
```

#### Action Space

<table>
<tr><th>Variant</th><th>Dimensions</th> <th>Parameters</th></tr>
<tr> 
<td> 

`v1` 

</td> 
<td> 

`ℝ ⁵ ∈ [-1.0; 1.0]`

</td>
<td>

```python
["Wn0", "Wp", "Wn2", "Wn1", "Wn3"]
```

</td>
</tr>
</table>

```python
# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (5 , )
              , dtype = np.float32
              , )
```


