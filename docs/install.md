## Installation and Setup

### Requirements

After installing [AC²E](https://github.com/matthschw/ace) and
[HAC²E](https://github.com/AugustUnderground/hace), including all dependencies,
there are multiple ways to configure `gace` and tell it, where it can find the
pdk, testbenches and machine learning models.

### Installation

```bash
$ pip install git+https://github.com/augustunderground/gace.git
```

Or clone and install a local copy

```bash
$ git clone https://github.com/augustunderground/gace.git
$ pip install .
```

### AC²E Backends

GAC²E, as the name suggests, depends on [AC²E](https://github.com/matthschw/ace). 

**Make sure AC²E is installed and functions properly before continuing!**

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

Where `<tech>` has to be a valid backend such as `gpdk180-1V8` for example.
