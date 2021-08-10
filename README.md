# Analog Design Gym

[gym](https://gym.openai.com/) environments for analog integrated circuit design.

## Installation

```bash
$ pip install . --use-feature=in-tree-build
```

## Requirements

Creating an environment requires NMOS and PMOS models as well as the
[analog-circuit-library](https://gitlab-forschung.reutlingen-university.de/schweikm/analog-circuit-library)
java package.

## Getting Started

After installing:

```python
import gym

env = gym.make( "gym_ad:symmetrical-amplifier-v0"
              , nmos_path   = "../models/xh035-nmos
              , pmos_path   = "../models/xh035-pmos
              , lib_path    = "/path/to/xfab/xh035/.../cadence/.../mos"
              , ckt_path    = "../libs"
              , jar_path    = "/path/to/edlab/eda/characterization-with-dependencies.jar"
              , )
```

Where `<nmos|pmos>_path` is the location of the corresponding `model.pt` and
input and output scaler `scale.X`.

## Environments

### Observation Space

All Amplifier environments share the same kind of observations of type `Dict`:

```json
{ target: { "a_0":       DC Gain
          , "ugbw":      Unity Gain Bandwidth
          , "pm":        Phase Margin
          , "gm":        Gain Margin
          , "sr_r":      Slew Rate rising
          , "sr_f":      Slew Rate falling
          , "vn_1Hz":    Signal to Noise 1Hz
          , "vn_10Hz":   Signal to Noise 10Hz
          , "vn_100Hz":  Signal to Noise 100Hz
          , "vn_1kHz":   Signal to Noise 1kHz
          , "vn_10kHz":  Signal to Noise 10kHz
          , "vn_100kHz": Signal to Noise 100kHz
          , "psrr_p":    Power Supply Rejection Ratio
          , "psrr_n":    Power Supply Rejection Ratio
          , "cmrr":      Common Mode Rejection Ratio
          , "v_il" 
          , "v_ih" 
          , "v_ol"
          , "v_oh" 
          , "i_out_min" 
          , "i_out_max"
          , "voff_stat": Statistical Offset
          , "voff_sys":  Systematic Offset
          , "A"
          , }
, performance: { ... }
, distance: { ... }
, waves: {  }
, }
```

Where the `performance` are the simulation / analyses results and `distance` is
`| target - performance |`.

### Symmetrical Amplifier

![sym](https://raw.githubusercontent.com/AugustUnderground/smacd2021-b4.4/master/notebooks/fig/sym.png)

#### With Reduced Action Space

Registered as `gym_ad:sym-amp-xh035-v0`.

##### Action Space

4 `gmoverid`s and `fug`s for each building block, `(, 8)` in total.

#### With Extended Action Space

**WIP**

Registered as `gym_ad:sym-amp-xh035-v1`.

##### Action Space

4 `gmoverid`s and `fug`s for each building block and branch currents `i1` and
`i2` which influence the mirror ratio. `(, 10)` in total.

### Miller Amplifier

**WIP**

Registered as `gym_ad:miller-amp-xh035-v0`.

![moa](https://raw.githubusercontent.com/AugustUnderground/smacd2021-b4.4/master/notebooks/fig/moa.png)
