A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/matthschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

## Availability Matrix

| Environment                                                                     | [xh035](https://gitlab-forschung.reutlingen-university.de/eda/ace-xh035-3v3) | [sky130](https://github.com/matthschw/ace-sky130-1V8) | [gpdk180](https://github.com/AugustUnderground/ace-gpdk180-1V8) | [xh018](https://gitlab-forschung.reutlingen-university.de/eda/ace-xh018-1v8) | [xt018](https://gitlab-forschung.reutlingen-university.de/eda/ace-xt018-1v8) | [ptm130](https://github.com/AugustUnderground/ace-ptm) |
|---------------------------------------------------------------------------------|:----------------------------------------------------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|:------------------------------------------------------:|
| [OP1](https://raw.githubusercontent.com/matthschw/ace/main/figures/op1.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP2](https://raw.githubusercontent.com/matthschw/ace/main/figures/op2.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP3](https://raw.githubusercontent.com/matthschw/ace/main/figures/op3.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP4](https://raw.githubusercontent.com/matthschw/ace/main/figures/op4.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP5](https://raw.githubusercontent.com/matthschw/ace/main/figures/op5.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP6](https://raw.githubusercontent.com/matthschw/ace/main/figures/op6.png)     |                                  `v0`, `v1`                                  |                       `v0`, `v1`                      |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP7](https://raw.githubusercontent.com/matthschw/ace/main/figures/op7.png)     |                                      NA                                      |                           NA                          |                                NA                               |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP8](https://raw.githubusercontent.com/matthschw/ace/main/figures/op8.png)     |                                  `v0`, `v1`                                  |                           NA                          |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP9](https://raw.githubusercontent.com/matthschw/ace/main/figures/op9.png)     |                                  `v0`, `v1`                                  |                           NA                          |                            `v0`, `v1`                           |                                      NA                                      |                                      NA                                      |                           NA                           |
| [OP10](https://raw.githubusercontent.com/matthschw/ace/main/figures/op10.png)   |                                      NA                                      |                           NA                          |                                NA                               |                                      NA                                      |                                      NA                                      |                           NA                           |
| [NAND4](https://raw.githubusercontent.com/matthschw/ace/main/figures/nand4.png) |                                     `v1`                                     |                          `v1`                         |                               `v1`                              |                                      NA                                      |                                      NA                                      |                           NA                           |
| [ST1](https://raw.githubusercontent.com/matthschw/ace/main/figures/st1.png)     |                                     `v1`                                     |                          `v1`                         |                               `v1`                              |                                      NA                                      |                                      NA                                      |                           NA                           |

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


