<h1 align="center">GAC²E</h1>

A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/matthschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

## Installation

```bash
$ pip install git+https://github.com/augustunderground/gace.git
```

Or clone and install a local copy

```bash
$ git clone https://github.com/augustunderground/gace.git
$ pip install .
```

See the [guide](#installation-and-setup) for detailed instructions on how to
setup the backends and machine learning models.

## Requirements

After installing [AC²E](https://github.com/matthschw/ace) and
[HAC²E](https://github.com/AugustUnderground/hace), including all dependencies,
there are multiple ways to configure `gace` and tell it, where it can find the
pdk, testbenches and machine learning models.

## Usage

```python
import gym

# Geometric design space and $HOME/.ace symlink or corresponding env vars
env = gym.make('gace:op2-sky130-v1')     # Symmetrical Amplifier in SkyWater 130nm

# Electrical design space and all kwargs
env = gym.make(                      'gace:op2-sky130-v1'    # OP2 in sky130-1V8
              , pdk_path           = '/path/to/tech'         # path to pdk
              , ckt_path           = '/path/to/op2'          # path to testbench
              , nmos_path          = '/path/to/models/nmos'  # path to nmos model
              , pmos_path          = '/path/to/models/pmos'  # path to pmos model
              , max_steps          = 200                     # Reset env after this many steps
              , target             = {}                      # Dict like 'perforamnce' below
              , design-constraints = {}                      # Override default constraints
              , random_target      = False                   # start close to target
              , noisy_target       = False                   # add some noise after each reset
              , data_log_path      = '/path/to/data/log'     # Write data after each episode
              , params_log_path    = '/path/to/param/log'    # Dump circuit state if NaN
              #, reltol             = 1e-3                    # ONLY FOR NAND4 AND ST1
              , )
```

The `design-constraints` dict supports the following fields:

```json
{ "cs":       "Poly Capacitance per μm^2"
, "rs":       "Sheet Resistance in Ω/□ "
, "i0":       "Bias Current in A"
, "vdd":      "Supply Voltage"
, "Wres":     "Resistor Width in m"
, "Mcap":     "Capacitance multiplier"
, "Rc_min":   "Minimum Compensation Resistor = 500Ω"
, "Rc_max":   "Minimum Compensation Resistor = 50000Ω"
, "Cc_min":   "Minimum Compensation Capacitor = 0.5pF"
, "Cc_max":   "Minimum Compensation Capacitor = 5pF"
, "w_min":    "Minimum width either in m or scaled to μm"
, "w_max":    "Maximum width either in m or scaled to μm"
, "l_min":    "Minimum length either in m or scaled to μm"
, "l_max":    "Maximum length either in m or scaled to μm"
, "gmid_min": "Minimum device efficiency"
, "gmid_max": "Maximum device efficiency"
, "fug_min":  "Minimum device speed"
, "fug_max":  "Maximum device speed"
}
```

Other fields should have no effect.

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

- [X] X-Fab XH035 350nm as `xh035`
- [X] SkyWater 130nm as `sky130`
- [X] GPDK 180nm
- [ ] PTM 130nm

### Action Spaces and Variants

Action Spaces in `v0` and `v1` are _continuous_ and implemented with
`gym.spaces.Box`. For further details, see the descriptions for specific
environments.

| Variant | Description                                      |
|---------|--------------------------------------------------|
| `v0`    | Electrical design space (`gmoverid`, `fug`, ...) |
| `v1`    | Geometric Design Space (`W`, `L`, ...)           |
| `v2`    | TBA                                              |

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

and `m` is a mask showing whether the performance was reached, i.e. `p > t`, in
which case `tanh` is applied so the reward doesn't increase infinitely.
Otherwise the loss is squared and negated.


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
<tr><th>Variant</th><th>Dimensions</th> <th>Description</th></tr>
<tr> 
<td> 
`v0` 
</td> 
<td> 
`ℝ ¹² ∈ [-1.0; 1.0]` 
</td>
<td>

```python
[ "gmid-cm1", "gmid-cm2", "gmid-cs1", "gmid-dp1"
, "fug-cm1", "fug-cm2", "fug-cs1", "fug-dp1" 
, "res", "cap", "i1", "i2" ]
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
[ "Ld", "Lcm1",  "Lcm2",  "Lcs",          "Lres"
, "Wd", "Wcm1",  "Wcm2",  "Wcs", "Wcap",  "Wres"
      , "Mcm11",          "Mcs"
      , "Mcm12" 
      , "Mcm13"                                 ]
```

</td>
</tr>
</table>

| Variant | Dimensions           | Description                                                                                                                     |
|---------|----------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹² ∈ [-1.0; 1.0]` | 4 `gmoverid`s and `fug`s for each building block,<br/> 1 resistance, 1 capacitance and the branch currents <br/> `i1` and `i2`. |
| `v1`    | `ℝ ¹⁵ ∈ [-1.0; 1.0]` | 4 `W`s and `L`s for each building block,<br/> geometric sizes for Rc and Cc as well as <br/> mirror rations `M1` and `M2`.      |

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

### Symmetrical Amplifier (OP2)

![op2](https://github.com/matthschw/ace/blob/main/figures/op2.png)

Registered as `gace:op2-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²⁰⁶ ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ²⁶⁶ ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ²⁹⁴ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (206 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (266 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (294 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions           | Description                                                                               |
|---------|----------------------|-------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁰ ∈ [-1.0; 1.0]` | 4 `gmoverid`s and `fug`s for each building <br/> block and branch currents `i1` and `i2`. |
| `v1`    | `ℝ ¹² ∈ [-1.0; 1.0]` | 4 `W`s and `L`s for each building block and <br/> mirror ratios `Mcm1` and `Mcm2`.        |

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (10 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (12 , )
              , dtype = np.float32
              , )
```

### Un-Symmetrical Amplifier (OP3)

![op3](https://github.com/matthschw/ace/blob/main/figures/op3.png)

Registered as `gace:op3-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ²⁴⁶ ∈ (-∞ ; ∞)` |
| `sky130`   | `ℝ ²⁶⁶ ∈ (-∞ ; ∞)` |
| `gpdk180`  | `ℝ ²⁹⁴ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (246 , )
              , dtype = np.float32
              , )

# sky130
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (266 , )
              , dtype = np.float32
              , )

# gpdk180
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (294 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions           | Description                                                                                                                                            |
|---------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹¹ ∈ [-1.0; 1.0]` | Same as _op2_ with an additional branch current. <br/> 4 `gmoverid`s and `fug`s for each building <br/> block and branch currents `i1`, `i2` and `i3`. |
| `v1`    | `ℝ ¹⁵ ∈ [-1.0; 1.0]` | Same as _op2_ with an additional mirror ratio. <br/> 4 `W`s and `L`s for each building block and <br/> mirror ratios `M1` `M2` and `M3`.               |

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (11 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (15 , )
              , dtype = np.float32
              , )
```

### Symmetrical Amplifier with Cascode (OP4)

![op4](https://github.com/matthschw/ace/blob/main/figures/op4.png)

Registered as `gace:op4-<tech>-<variant>`.

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

| Variant | Dimensions           | Description                                                                                                                 |
|---------|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁵ ∈ [-1.0; 1.0]` | 6 `gmoverid`s and `fug`s for each building <br/> block and branch currents `i1`, `i2` and `i3`.                             |
| `v1`    | `ℝ ¹⁸ ∈ [-1.0; 1.0]` | 6 `W`s and `L`s for each building block and <br/> reference transistor, as well as <br/> mirror ratios `M1`, `M2` and `M3`. |

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (15 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (18 , )
              , dtype = np.float32
              , )
```

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

| Variant | Dimensions           | Description                                                                                                                       |
|---------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁶ ∈ [-1.0; 1.0]` | 6 `gmoverid`s and `fug`s for each building <br/> block and branch currents `i1`, `i2`, `i3` and `i4`.                             |
| `v1`    | `ℝ ²² ∈ [-1.0; 1.0]` | 6 `W`s and `L`s for each building block and <br/> reference transistor, as well as <br/> mirror ratios `M1`, `M2`, `M3` and `M4`. |

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

### Miller Amplifier without R and C (OP6)

![op6](https://github.com/matthschw/ace/blob/main/figures/op6.png)

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

| Variant | Dimensions           | Description                                                                                                                |
|---------|----------------------|----------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁴ ∈ [-1.0; 1.0]` | 4 `gmoverid`s and `fug`s for each building <br/> block, the weird transistors and the branch <br/> currents `i1` and `i2`. |
| `v1`    | `ℝ ¹⁸ ∈ [-1.0; 1.0]` | 6 `W`s and `L`s for each building block and <br/> the weird transistors plus the mirror ratios <br/> `M1` and `M2`.        |

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

### TBA (OP7)

![op7](https://github.com/matthschw/ace/blob/main/figures/op7.png)

**. . .**

### WORK IN PROGRESS (OP8)

![op8](https://github.com/matthschw/ace/blob/main/figures/op8.png)

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

| Variant | Dimensions           | Description                                                                                   |
|---------|----------------------|-----------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁶ ∈ [-1.0; 1.0]` | 6 `gmoverid`s and `fug`s for each building <br/> block and the branch currents `i1` and `i2`. |
| `v1`    | `ℝ ²¹ ∈ [-1.0; 1.0]` | 6 `W`s and `L`s for each building block and <br/> plus a bunch of mirror ratios.              |

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

### WORK IN PROGRESS (OP9)

![op9](https://github.com/matthschw/ace/blob/main/figures/op9.png)

_Will be_ registered as `gace:op9-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions         |
|------------|--------------------|
| `xh035`    | `ℝ ³⁷¹ ∈ (-∞ ; ∞)` |

```python
# xh035
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (371 , )
              , dtype = np.float32
              , )
```

#### Action Space

| Variant | Dimensions           | Description                                                                      |
|---------|----------------------|----------------------------------------------------------------------------------|
| `v0`    | `ℝ ²² ∈ [-1.0; 1.0]` | 8 `gmoverid`s and `fug`s for each building <br/> block and 6 branch currents.    |
| `v1`    | `ℝ ²⁷ ∈ [-1.0; 1.0]` | 8 `W`s and `L`s for each building block and <br/> plus a bunch of mirror ratios. |

```python
# v0 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (22 , )
              , dtype = np.float32
              , )

# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (27 , )
              , dtype = np.float32
              , )
```

### TBA (OP10)

![op10](https://github.com/matthschw/ace/blob/main/figures/op10.png)

**. . .**

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

| Variant | Dimensions          | Description                        |
|---------|---------------------|------------------------------------|
| `v1`    | `ℝ ⁵ ∈ [-1.0; 1.0]` | 5 `W`s for each gate in the chain. |

```python
# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (5 , )
              , dtype = np.float32
              , )
```

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

![st1](https://github.com/matthschw/ace/blob/main/figures/st1.png)

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

| Variant | Dimensions          | Description                               |
|---------|---------------------|-------------------------------------------|
| `v1`    | `ℝ ⁶ ∈ [-1.0; 1.0]` | 6 `W`s for each device in schmit trigger. |

```python
# v1 action space
gym.spaces.Box( low   = -1.0
              , high  = 1.0
              , shape = (6 , )
              , dtype = np.float32
              , )
```

## Installation and Setup

### AC²E Backends

#### Symlink

One way is to create a symlink from the `resource` directory of the AC²E
repository, which contains all the backends as git submodules, to `~/.ace`.
`gace` will automatically look there if `pdk_path` and `ckt_path` are not
specified.

```bash
$ ln -s /path/to/ace/resource $HOME/.ace
```

It should look sort of like this:

```
$HOME/.ace
├── sky130-1V8
│   ├── LICENSE
│   ├── nand4
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op1
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op2
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op3
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op4
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op5
│   │   ├── input.scs
│   │   └── properties.json
│   ├── op6
│   │   ├── input.scs
│   │   └── properties.json
│   ├── pdk
│   │   ├── cells
│   │   │   ├── nfet_01v8
│   │   │   │   ├── sky130_fd_pr__nfet_01v8__mismatch.corner.scs
│   │   │   │   ├── sky130_fd_pr__nfet_01v8__tt.corner.scs
│   │   │   │   └── sky130_fd_pr__nfet_01v8__tt.pm3.scs
│   │   │   └── pfet_01v8
│   │   │       ├── sky130_fd_pr__pfet_01v8__mismatch.corner.scs
│   │   │       ├── sky130_fd_pr__pfet_01v8__tt.corner.scs
│   │   │       └── sky130_fd_pr__pfet_01v8__tt.pm3.scs
│   │   ├── models
│   │   │   ├── all.scs
│   │   │   ├── corners
│   │   │   │   └── tt
│   │   │   │       └── nofet.scs
│   │   │   ├── parameters
│   │   │   │   └── lod.scs
│   │   │   └── sky130.scs
│   │   ├── README.md
│   │   └── tests
│   │       ├── nfet_01v8_tt.scs
│   │       └── pfet_01v8_tt.scs
│   ├── nmos -> /path/to/sky130-nmos
│   ├── pmos -> /path/to/sky130-pmos
│   ├── README.md
│   └── st1
│       ├── input.scs
│       └── properties.json
└── xh035-3V3
    ├── LICENSE
    ├── nand4
    │   ├── input.scs
    │   └── properties.json
    ├── op1
    │   ├── input.scs
    │   └── properties.json
    ├── op2
    │   ├── input.scs
    │   └── properties.json
    ├── op3
    │   ├── input.scs
    │   └── properties.json
    ├── op4
    │   ├── input.scs
    │   └── properties.json
    ├── op5
    │   ├── input.scs
    │   └── properties.json
    ├── op6
    │   ├── input.scs
    │   └── properties.json
    ├── op8
    │   ├── input.scs
    │   └── properties.json
    ├── op9
    │   ├── input.scs
    │   └── properties.json
    ├── pdk -> /path/to/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos
    ├── nmos -> /path/to/xh035-nmos
    ├── pmos -> /path/to/xh035-pmos
    ├── README.md
    └── st1
        ├── input.scs
        └── properties.json
```

#### Environment Variables

Alternatively you can set environment variables, telling `gace` where to find
the pdk and testbenches.

```bash
$ export ACE_BACKEND=/path/to/ace/resource
$ export ACE_PDK=/path/to/ace/resource/<tech>/pdk
```

Where `<tech>` has to be a valid backend such as `xh035-3V3` for example.

### Machine Learning Models for v0 Environments

The models used for `v0` envs are trained with
[precept](https://github.com/electronics-and-drives/precept) and _must_ have
the following mapping:

```
[ gmoverid          [ log₁₀(idoverw)
, log₁₀(fug)   ↦    , L
, Vds               , log₁₀(gdsoverw)
, Vbs ]             , Vgs ]
```

The paths (`nmos_path`, `pmos_path`) _must_ be structured like this:

```
nmos_path
├── model.ckpt  # Optional
├── model.pt    # TorchScript model produced by precept
├── scale.X     # Scikit MinMax Scaler for inputs
└── scale.Y     # Scikit MinMax Scaler for outputs
```

The `model.pt` _must_ be a
[torchscript](https://pytorch.org/tutorials/recipes/torchscript_inference.html)
module adhering to the specified input and output dimensions. The scalers
`scale.<X|Y>` _must_ be
[MinMaxScalers](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
dumped with joblib.

#### Symlink

Instead of passing `nmos_path` and `pmos_path` it is also possible to
copy/symlink the above mentioned directory into the ACE home directory:

```bash
$ ln -s /path/to/model/nmos $HOME/.ace/<tech>/nmos
$ ln -s /path/to/model/pmos $HOME/.ace/<tech>/nmos
```

Where `<tech>` has to be a valid backend such as `xh035-3V3` for example.

## Known Issues, Debugging and Testing

Whenever the environment produces observations that contain `NaN` a
`parameters-<timestamp>.json` file will be produced. This can be used with
[HAC²E](https://github.com/AugustUnderground/hace) to load that particular
state and analyze what went wrong.

Currently we cope with `NaN`s by using numpy's 
[nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html) 
function. Be aware, such values can cause problems on GPU.

Test can be run, if [pytest](https://pytest.org) is installed, by calling
`pytest` in the root of this repository, which takes about ~5 min.

**NOTE:** The tests only work if `$HOME/.ace` exists and contains valid
[AC²E](https://github.com/matthschw/ace) backends including the pdk.

```bash
$ pytest
```

## TODO

- [X] fix reward function
- [X] remove target tolerance 
- [X] adjust info key for observations
- [X] set done when met mask true for all
- [X] new Env with geometrical action space
- [X] restructure to fit ace
- [ ] new Env with sim mask as action
- [ ] demo Agent
- [X] handle `NaN`s better
- [ ] find better limit/range for obs space
- [X] add skywater envs
- [X] add gpdk envs
- [ ] add installer script
- [ ] add reset counter, gradually increase noise/target distance
