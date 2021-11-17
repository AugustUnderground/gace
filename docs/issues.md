## Known Issues, Debugging and Testing

Whenever the environment produces observations that contain `NaN` a
`parameters-<timestamp>.json` file will be produced. This can be used with
[HAC²E](https://github.com/AugustUnderground/hace) to load that particular
state and analyze what went wrong.

Currently we cope with `NaN`s by using numpy's 
[nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html) 
function. Be aware, such values can cause problems on GPU.

Test can be run, if [pytest](https://pytest.org) is installed, by calling
`pytest` in the root of this repository, which takes about ~5 min.

**NOTE:** The tests only work if `$HOME/.ace` exists and contains valid
[AC²E](https://github.com/matthschw/ace) backends including the pdk.

```bash
$ pytest
```

## TODO

- [X] fix reward function
- [X] remove target tolerance 
- [X] adjust info key for observations
- [X] set done when met mask true for all
- [X] new Env with geometrical action space
- [X] restructure to fit ace
- [ ] new Env with sim mask as action
- [ ] demo Agent
- [X] handle `NaN`s better
- [ ] find better limit/range for obs space
- [X] add skywater envs
- [X] add gpdk envs
- [ ] add installer script
- [ ] add reset counter, gradually increase noise/target distance
- [ ] document which branch current is which
- [ ] fix `v0` issues with `gpdk180` (probably ML models borked)
