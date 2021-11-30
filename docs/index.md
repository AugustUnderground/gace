A collection of [gym](https://gym.openai.com/) environments for analog 
integrated circuit design, based on [AC²E](https://github.com/matthschw/ace) /
[HAC²E](https://github.com/AugustUnderground/hace).

## Availability Matrix

<table>
<tr>
<th></th>
<th>
<a href="https://gitlab-forschung.reutlingen-university.de/eda/ace-xh035-3v3">xh035</a>
</th>
<th>
<a href="https://gitlab-forschung.reutlingen-university.de/eda/ace-xh018-1v8">xh018</a>
</th>
<th>
<a href="https://gitlab-forschung.reutlingen-university.de/eda/ace-xt018-1v8">xt018</a>
</th>
<th>
<a href="https://github.com/matthschw/ace-sky130-1V8">sky130</a>
</th>
<th>
<a href="https://github.com/AugustUnderground/ace-gpdk180-1V8">gpdk180</a>
</th>
<th>
<a href="https://github.com/AugustUnderground/ace-ptm">ptm130</a>
</th>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op1.png">op1</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op2.png">op2</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op3.png">op3</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op4.png">op4</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op5.png">op5</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op6.png">op6</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op7.png">op7</a>
</th>
<td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op8.png">op8</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op9.png">op9</a>
</th>
<td>v0, v1</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>v0, v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/op10.png">op10</a>
</th>
<td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/nand4.png">nand4</a>
</th>
<td>v1</td> <td>NA</td> <td>NA</td> <td>v1</td> <td>v1</td> <td>NA</td>
</tr>
<tr>
<th>
<a href="https://raw.githubusercontent.com/matthschw/ace/main/figures/st1.png">st1</a>
</th>
<td>v1</td> <td>NA</td> <td>NA</td> <td>v1</td> <td>v1</td> <td>NA</td>
</tr>
</table>

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
    + [X] [NAND4: Inverter Chain](./nd4.md)
    + [X] [ST1: Schmitt Trigger](./st1.md)
- [Issues](./issues.md)


