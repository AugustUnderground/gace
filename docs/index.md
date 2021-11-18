A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/matthschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

## Availability Matrix

| Environment |   `xh035`  |  `sky130`  |  `gpdk180` | `xh018` | `ptm130` |
|-------------|:----------:|:----------:|:----------:|:-------:|:--------:|
| **OP1**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP2**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP3**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP4**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP5**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP6**     | `v0`, `v1` | `v0`, `v1` | `v0`, `v1` |    NA   |    NA    |
| **OP7**     |     NA     |     NA     |     NA     |    NA   |    NA    |
| **OP8**     | `v0`, `v1` |     NA     | `v0`, `v1` |    NA   |    NA    |
| **OP9**     | `v0`, `v1` |     NA     | `v0`, `v1` |    NA   |    NA    |
| **OP10**    |     NA     |     NA     |     NA     |    NA   |    NA    |
| **NAND4**   |    `v1`    |    `v1`    |    `v1`    |    NA   |    NA    |
| **ST1**     |    `v1`    |    `v1`    |    `v1`    |    NA   |    NA    |

## Table of Contents

- [Installation](./install.md)
- [Basic Usage](./usage.md)
- **Environments:**
- [Single Ended Operational Amplifiers](./op0.md)
    + [X] [OP1: Miller Operational Amplifier](./op1.md)
    + [X] [OP2: Symmetrical Amplifier](./op2.md)
    + [X] [OP3: Un-Symmetrical Amplifier](./op3.md)
    + [X] [OP4: Symmetrical Cascode Amplifier](./op4.md)
    + [X] [OP5: Un-Symmetrical Cascode Amplifier](./op5.md)
    + [X] [OP6: Miller Amplifier w/o Passives](./op6.md)
    + [ ] [OP7: TBA](./op7.md)
    + [X] [OP8: Amplifier with Wideswing Current Mirror](./op8.md)
    + [X] [OP9: Amplifier with Cascode Wideswing Current Mirror](./op9.md)
    + [ ] [OP10: TBA](./op10.md)
- Other Environments
    + [X] [NAND4: Inverter Chain](./nand4.md)
    + [X] [ST1: Schmitt Trigger](./st1.md)
- [Issues](./issues.md)


