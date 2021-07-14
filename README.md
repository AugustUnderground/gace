# Analog Design Gym

[gym](https://gym.openai.com/) environments for analog integrated circuit design.

## Installation

```bash
$ pip install . --use-feature=in-tree-build
```

## Requirements

Creating an environment requires NMOS and PMOS models as well as corresponding
spice models.

### Spice Simulation Models

[PTM](http://ptm.asu.edu/) spice libraries for different technology nodes are
freely available.

### Machine Learning Models

Generate data with [pyrdict](https://github.com/AugustUnderground/pyrdict).
Train models with [precept](https://github.com/electronics-and-drives/precept).

## Test

After obtaining models and spice libraries, reference them in `test/test.py`
and run the script.

```bash
$ cd test/
$ python test.py
```

## Getting Started

```python
import gym

env = gym.make( "gym_ad:symmetrical-amplifier-v0"
              , nmos_prefix = "../models/90nm-nmos"
              , pmos_prefix = "../models/90nm-pmos"
              , lib_path    = "../libs/90nm_bulk.lib"
              , )
```

Where `<nmos|pmos>_prefix` is the location of `.pt` and input and output scaler.

## Environments

### Symmetrical Amplifier

Registered as `gym_ad:symmetrical-amplifier-v0`.

![symamp](https://raw.githubusercontent.com/AugustUnderground/smacd2021-b4.4/master/notebooks/fig/sym.png)

#### Action Space



#### Observation Space

### Miller Amplifier

