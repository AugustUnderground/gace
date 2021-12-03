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
r = ∑ ( (tanh(| l |) · m) + (- l² · (1 - m)) )
```

Where `l` is the vector with normalized losses for each performance `p` and
target value `t`

```
    | t - p |
l = ---------
        t
```

and `m` is a mask showing whether the performance was reached, i.e. 

```
(t - (t * reltol)) < p < (t + (t * reltol))
``` 

meaning the performance `p` is within a tolerance band, in
which case `tanh` is applied so the reward doesn't increase infinitely.
Otherwise the loss is squared and negated.

### Schmitt Trigger (ST1)

![st1](https://raw.githubusercontent.com/matthschw/ace/main/figures/st1.png)

Registered as `gace:st1-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions    |
|------------|---------------|
| `xh035`    | `ℝ¹²∈(-∞ ;∞)` |
| `xh018`    | `ℝ¹²∈(-∞ ;∞)` |
| `xt018`    | `ℝ¹²∈(-∞ ;∞)` |
| `sky130`   | `ℝ¹²∈(-∞ ;∞)` |
| `gpdk180`  | `ℝ¹²∈(-∞ ;∞)` |

For details see the `output-parameters` field of the `info` dictionary
returned by `step()`.

```python
# xh035, xh018, xt018, sky130, gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (12 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions      | Parameters                                   |
|---------|-----------------|----------------------------------------------|
| `v1`    | `ℝ⁶∈[-1.0;1.0]` | `["Wp0", "Wn0", "Wp2", "Wp1", "Wn2", "Wn1"]` |
| `v3`    | `ℝ⁶∈[0,1,2]`    | `["Wp0", "Wn0", "Wp2", "Wp1", "Wn2", "Wn1"]` |

```python
# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (6 , )
              , dtype = np.float32
              , )

# v3 action space
gym.spaces.MultiDiscrete( list(repeat(3, 6))
                        , dtype = np.int32
                        , )
```

