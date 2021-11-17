A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/matthschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

### AC²E Backends

It is planned to support _all_ AC²E backends in the future. For now the
following are (partially) implemented.

- [X] X-Fab XH035 350nm as `xh035`
- [ ] X-Fab XH018 180nm as `xh018`
- [ ] X-Fab XT018 180nm as `xt035`
- [X] SkyWater 130nm as `sky130`
- [X] GPDK 180nm as `gpdk180`
- [ ] GPDK 90nm as `gpdk90`
- [ ] PTM 130nm

## Table of Content

- [Installation](./install.md)
- [Basic Usage](./usage.md)
- [Single Ended Operational Amplifiers](./op0.md)
    + [X] [Miller Operational Amplifier (OP1)](./op1.md)
    + [X] [Symmetrical Amplifier (OP2)](./op2.md)
    + [X] [Un-Symmetrical Amplifier (OP3)](./op3.md)
    + [X] [Symmetrical Cascode Amplifier (OP4)](./op4.md)
    + [X] [Un-Symmetrical Cascode Amplifier (OP5)](./op5.md)
    + [X] [Miller Amplifier w/o Passives (OP6)](./op6.md)
    + [ ] [TBA (OP7)](./op7.md)
    + [X] [Amplifier with Wideswing Current Mirror (OP8)](./op8.md)
    + [X] [Amplifier with Cascode Wideswing Current Mirror (OP9)](./op9.md)
    + [ ] [TBA (OP10)](./op10.md)
- Other Environments
    + [X] [NAND4](./nand4.md)
    + [X] [ST1](./st1.md)
- [Issues](./issues.md)
