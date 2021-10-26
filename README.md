# Analog Design Gym

A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/mattschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

## Installation

```bash
$ pip install git+https://github.com/augustunderground/gym-analog-design.git
```

## Requirements

Creating an environment requires trained NMOS and PMOS models and HAC²E
including all dependencies.

## Getting Started

After installing:

```python
import gym

env = gym.make(             'gym_ad:op2-xh035-v0'   # Only working env right now
              , pdk-path  = '/path/to/tech'         # path to xfab pdk
              , ckt-path  = '/path/to/op2'          # path to testbench
              , nmos-path = '/path/to/models/nmos'  # path to nmos model
              , pmos-path = '/path/to/models/pmos'  # paht to pmos model
              , random-target = False )             # start close to target
```

Where `<nmos|pmos>_path` is the location of the corresponding `model.pt`.

## Environments

### Observation Space

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

### Action Spaces

All actions spaces are _continuous_ and implemented with `gym.spaces.Box`. For
further details, see the descriptions for specific environments.

### Variations

| Version | Description                                      |
|---------|--------------------------------------------------|
| `v0`    | Electrical design space (`gmoverid`, `fug`, ...) |
| `v1`    | Geometric Design Space (`W`, `L`, ...)           |
| `v2`    | TBA                                              |

### Miller Amplifier (OP1)

![op1](https://github.com/matthschw/ace/blob/main/figures/op1.png)

Registered as `gym_ad:op1-xh035-vX`.

#### Action Space 

| Version | <div style="width:290px">Domain</div> | Description                                                                                                          |
|---------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹² ∈ [-1.0; 1.0]`                           | 4 `gmoverid`s and `fug`s for each building block, 1 resistance, 1 capacitance and the branch currents `i1` and `i2`. |

```python
# v0 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (12 , )
              , dtype = np.float32 
              , )
```

#### Observation Space

Continuous `ℝ ¹⁵⁰ ∈ (-∞ ; ∞)`:

```python
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (150 , )
              , dtype = np.float32
              , )
```

### Symmetrical Amplifier (OP2)

![op2](https://github.com/matthschw/ace/blob/main/figures/op2.png)

Registered as `gym_ad:op2-xh035-v0`.

#### Action Space

| Version | Domain               | Description                                                                         |
|---------|----------------------|-------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁰ ∈ [-1.0; 1.0]` | 4 `gmoverid`s and `fug`s for each building block and branch currents `i1` and `i2`. |
| `v1`    | `ℝ ¹² ∈ [-1.0; 1.0]` | 4 `W`s and `L`s for each building block and mirror ratios `Mcm1` and `Mcm2`.        |

```python
# v0 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (10 , )
              , dtype = np.float32 
              , )

# v1 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (12 , )
              , dtype = np.float32 
              , )
```

#### Observation Space

Continuous `ℝ ²⁴⁵ ∈ (-∞ ; ∞)`:

```python
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (245 , )
              , dtype = np.float32
              , )
```

### Un-Symmetrical Amplifier (OP3)

![op3](https://github.com/matthschw/ace/blob/main/figures/op3.png)

Registered as `gym_ad:op3-xh035-v0`.

#### Action Space

| Version | Domain               | Description                                                                                                                                |
|---------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹¹ ∈ [-1.0; 1.0]` | Same as _op2_ with an additional branch current. 4 `gmoverid`s and `fug`s for each building block and branch currents `i1`, `i2` and `i3`. |
| `v1`    | `ℝ ¹⁵ ∈ [-1.0; 1.0]` | Same as _op2_ with an additional mirror ratio. 4 `W`s and `L`s for each building block and mirror ratios `M1` `M2` and `M3`.               |

```python
# v0 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (11 , )
              , dtype = np.float32 
              , )

# v1 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (15 , )
              , dtype = np.float32 
              , )
```

#### Observation Space

Same as _op2_, continuous `ℝ ²⁴⁵ ∈ (-∞ ; ∞)`:

```python
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (245 , )
              , dtype = np.float32
              , )
```

### Symmetrical Amplifier with Cascode (OP4)

![op4](https://github.com/matthschw/ace/blob/main/figures/op4.png)

Registered as `gym_ad:op4-xh035-v0`.

#### Action Space

| Version | Domain               | Description                                                                               |
|---------|----------------------|-------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁵ ∈ [-1.0; 1.0]` | 6 `gmoverid`s and `fug`s for each building block and branch currents `i1`, `i2` and `i3`. |

```python
# v0 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (15 , )
              , dtype = np.float32 
              , )
```

#### Observation Space

Continuous `ℝ ²⁸⁴ ∈ (-∞ ; ∞)`:

```python
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (284 , )
              , dtype = np.float32
              , )
```

### Un-Symmetrical Amplifier with Cascode (OP5)

<b align="center">UNDER CONSTRUCTION</b>

### Miller Amplifier without R and C (OP6)

![op6](https://github.com/matthschw/ace/blob/main/figures/op6.png)

Registered as `gym_ad:op6-xh035-v0`.

#### Action Space

| Version | Domain               | Description                                                                                                    |
|---------|----------------------|----------------------------------------------------------------------------------------------------------------|
| `v0`    | `ℝ ¹⁴ ∈ [-1.0; 1.0]` | 4 `gmoverid`s and `fug`s for each building block, the weird transistors and the branch currents `i1` and `i2`. |

```python
# v0 action space
gym.spaces.Box( low = -1.0
              , high = 1.0
              , shape = (14 , )
              , dtype = np.float32 
              , )
```

#### Observation Space

Same as _op1_, continuous `ℝ ¹⁵⁰ ∈ (-∞ ; ∞)`:

```python
gym.spaces.Box( low   = -np.inf
              , high  = np.inf
              , shape = (150 , )
              , dtype = np.float32
              , )
```

## Debugging

Whenever the environment produces observations that contain `NaN` a
`parameters-<timestamp>.json` file will be produced. This can be used with
[HAC²E](https://github.com/AugustUnderground/hace) to load that particular
state and analyze what went wrong.

## TODO

- [X] fix reward function
- [X] remove target tolerance 
- [X] adjust info key for observations
- [X] set done when met mask true for all
- [ ] new Env with geometrical action space
- [ ] new Env with sim mask as action
- [ ] demo Agent
