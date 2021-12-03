## Single Ended OpAmp Environments

### Observation Spaces and Technologies

The observation space includes the current perforamnce, the specified target,
the normalized distance between target and current performance (`| t - p | / t`) 
and operating point characteristics for all devices in the circuit.

```json
{ "performance": { "a_0":         "DC Gain"
                 , "ugbw":        "Unity Gain Bandwidth"
                 , "pm":          "Phase Margin"
                 , "gm":          "Gain Margin"
                 , "sr_r":        "Slew Rate rising"
                 , "sr_f":        "Slew Rate falling"
                 , "vn_1Hz":      "Output-referred noise density @ 1Hz"
                 , "vn_10Hz":     "Output-referred noise density @ 10Hz"
                 , "vn_100Hz":    "Output-referred noise density @ 100Hz"
                 , "vn_1kHz":     "Output-referred noise density @ 1kHz"
                 , "vn_10kHz":    "Output-referred noise density @ 10kHz"
                 , "vn_100kHz":   "Output-referred noise density @ 100kHz"
                 , "psrr_p":      "Power Supply Rejection Ratio"
                 , "psrr_n":      "Power Supply Rejection Ratio"
                 , "cmrr":        "Common Mode Rejection Ratio"
                 , "v_il":        "Input Low"
                 , "v_ih":        "Input High"
                 , "v_ol":        "Output Low"
                 , "v_oh":        "Output High"
                 , "i_out_min":   "Minimum output current"
                 , "i_out_max":   "Maximum output current"
                 , "overshoot_r": "Slew rate overswing rising"
                 , "overshoot_f": "Slew rate overswing falling"
                 , "voff_stat":   "Statistical Offset"
                 , "voff_sys":    "Systematic Offset"
                 , "A":           "Area" }
, "target":      { "Same keys as 'performance'": "..." }
, "distance":    { "Same keys as 'performance'": "..." }
, "state":       { "electrical characteristics": "..." } }
```

### Action Spaces and Variants

Action Spaces in `v0` and `v1` are _continuous_ and implemented with
`gym.spaces.Box`. For further details, see the descriptions for specific
environments.

| Variant | Description                                                          |
|---------|----------------------------------------------------------------------|
| `v0`    | Absolute Continuous Electrical design space (`gmoverid`, `fug`, ...) |
| `v1`    | Absolute Continuous Geometric Design Space (`W`, `L`, ...)           |
| `v2`    | Relative Discrete Electrical design space (`gmoverid`, `fug`, ...)   |
| `v3`    | Relative Discrete Geometric Design Space (`W`, `L`, ...)             |

### Reward

By default the overall reward `r` is calculated based on individual performance
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

and `m` is a mask showing whether the performance was reached, i.e. `p > t`, in
which case `tanh` is applied so the reward doesn't increase infinitely.
Otherwise the loss is squared and negated.

**This behaviour can be overridden** by passing a `custom_reward` function to
the environment constructor (see [usage](./usage.md)).
