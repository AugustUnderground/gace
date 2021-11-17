### TBA (OP7)

![op7](https://raw.githubusercontent.com/matthschw/ace/main/figures/op7.png)

_Will be_ registered as `gace:op7-<tech>-<variant>`.

#### Observation Space

| Technology | Dimensions   |
|------------|--------------|
| `...`      | `ℝ ∈(-∞ ;∞)` |

```python
# <tech>
gym.spaces.Box( ... )
```

#### Action Space


| Variant | Dimensions      | Parameters |
|---------|-----------------|------------|
| `v0`    | `ℝ ∈[-1.0;1.0]` | `[]`       |
| `v1`    | `ℝ ∈[-1.0;1.0]` | `[]`       |

```python
# v0 action space
gym.spaces.Box( ... )

# v1 action space
gym.spaces.Box( ... )
```

