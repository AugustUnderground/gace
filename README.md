<h1 align="center">GAC²E</h1>

A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/electronics-and-drives/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

**Make sure to read the**
[documentation](https://augustunderground.github.io/gace/) for installation and
setup instructions.

## Getting Started

```bash
$ pip install git+https://github.com/augustunderground/gace.git
```

Or clone and install a local copy

```bash
$ git clone https://github.com/augustunderground/gace.git
$ pip install .
```

```python
import gym

env = gym.make('gace:op2-sky130-v1')
```

