## Schmitt Trigger Environments

### Observation Spaces and Technologies

The observation space includes the current perforamnce, the specified target,
the normalized distance between target and current performance (`| t - p | / t`) 
and operating point characteristics for all devices in the circuit.

```json
{ "v_il":  "Threshold Vdd/2 - delta"
, "v_ih":  "Threshold Vdd/2 + delta"
, "t_phl": "Propagation Delay"
, "t_plh": "Propagation Delay" 
, "target":   { "Same keys as 'performance'": "..." }
, "distance": { "Same keys as 'performance'": "..." } }
```

Since there are only comparatively few observations, they are identical across
technologies.

### Action Spaces

Action Spaces in `v1` are _continuous_ and implemented with
`gym.spaces.Box`. For further details, see the descriptions for specific
environments.  For now, only `v1` is implemented for Schmitt Trigger
Environments.

### Reward

The overall reward `r` is calculated based on individual performance
parameters:

```
r = - ∑ ( | t - p | / t )
```

### Schmitt Trigger (ST1)

![st1](https://raw.githubusercontent.com/matthschw/ace/main/figures/st1.png)

Registered as `gace:st1-<tech>-<variant>`.

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

`ℝ ⁶ ∈ [-1.0; 1.0]`

</td>
<td>

```python
["Wp0", "Wn0", "Wp2", "Wp1", "Wn2", "Wn1"]
```

</td>
</tr>
</table>

```python
# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (6 , )
              , dtype = np.float32
              , )
```


