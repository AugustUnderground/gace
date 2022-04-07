# Function for checking custom environments.

from .util import func
from .util import target
from .util import render
from .util.test import check_env
from .envs.vec import vector_make, vector_make_same
from .util.func import scale_value, unscale_value
## Environment Variants:
#
# | Variant | Description                                |
# |---------+--------------------------------------------|
# | v0      | Electrical Action Space (gmoverid and fug) |
# | v1      | Geometric Action Space (W, L and M)        |
# | v2      | TBA                                        |
#
# Technolgies:
#   - X-Fab 350nm: xh035-3V3
#   - X-Fab 180nm: xh018-1V8
#   - X-Fab 180nm: xt018-1V8
#   - SkyWater 130nm: sky130-1V8
#   - GPDK 180nm: gpdk180-1V2
#   - TODO PTM 130nm: ptm130-1V2

from gym.envs.registration import register

## AC²E: OP1 - Miller Amplifier
register( id          = 'op1-xh035-v0'
        , entry_point = 'gace.envs:OP1XH035V0Env'
        , )

register( id          = 'op1-xh035-v1'
        , entry_point = 'gace.envs:OP1XH035V1Env'
        , )

register( id          = 'op1-xh035-v3'
        , entry_point = 'gace.envs:OP1XH035V3Env'
        , )

#register( id          = 'op1-xh018-v0'
#        , entry_point = 'gace.envs:OP1XH018V0Env'
#        , )
#
#register( id          = 'op1-xh018-v1'
#        , entry_point = 'gace.envs:OP1XH018V1Env'
#        , )
#
#register( id          = 'op1-xh018-v3'
#        , entry_point = 'gace.envs:OP1XH018V3Env'
#        , )
#
#register( id          = 'op1-xt018-v0'
#        , entry_point = 'gace.envs:OP1XT018V0Env'
#        , )
#
#register( id          = 'op1-xt018-v1'
#        , entry_point = 'gace.envs:OP1XT018V1Env'
#        , )

#register( id          = 'op1-xt018-v3'
#        , entry_point = 'gace.envs:OP1XT018V3Env'
#        , )

register( id          = 'op1-sky130-v0'
        , entry_point = 'gace.envs:OP1SKY130V0Env'
        , )

register( id          = 'op1-sky130-v1'
        , entry_point = 'gace.envs:OP1SKY130V1Env'
        , )

register( id          = 'op1-sky130-v3'
        , entry_point = 'gace.envs:OP1SKY130V3Env'
        , )

register( id          = 'op1-gpdk180-v0'
        , entry_point = 'gace.envs:OP1GPDK180V0Env'
        , )

register( id          = 'op1-gpdk180-v1'
        , entry_point = 'gace.envs:OP1GPDK180V1Env'
        , )

register( id          = 'op1-gpdk180-v3'
        , entry_point = 'gace.envs:OP1GPDK180V3Env'
        , )

## AC²E: OP2 - Symmetrical Amplifier
register( id          = 'op2-xh035-v0'
        , entry_point = 'gace.envs:OP2XH035V0Env'
        , )

register( id          = 'op2-xh035-v1'
        , entry_point = 'gace.envs:OP2XH035V1Env'
        , )

register( id          = 'op2-xh035-v2'
        , entry_point = 'gace.envs:OP2XH035V2Env'
        , )

register( id          = 'op2-xh035-v3'
        , entry_point = 'gace.envs:OP2XH035V3Env'
        , )

#register( id          = 'op2-xh018-v0'
#        , entry_point = 'gace.envs:OP2XH018V0Env'
#        , )
#
#register( id          = 'op2-xh018-v1'
#        , entry_point = 'gace.envs:OP2XH018V1Env'
#        , )
#
#register( id          = 'op2-xh018-v3'
#        , entry_point = 'gace.envs:OP2XH018V3Env'
#        , )
#
#register( id          = 'op2-xt018-v0'
#        , entry_point = 'gace.envs:OP2XT018V0Env'
#        , )
#
#register( id          = 'op2-xt018-v1'
#        , entry_point = 'gace.envs:OP2XT018V1Env'
#        , )

#register( id          = 'op2-xt018-v3'
#        , entry_point = 'gace.envs:OP2XT018V3Env'
#        , )

register( id          = 'op2-sky130-v0'
        , entry_point = 'gace.envs:OP2SKY130V0Env'
        , )

register( id          = 'op2-sky130-v1'
        , entry_point = 'gace.envs:OP2SKY130V1Env'
        , )

register( id          = 'op2-sky130-v3'
        , entry_point = 'gace.envs:OP2SKY130V3Env'
        , )

register( id          = 'op2-gpdk180-v0'
        , entry_point = 'gace.envs:OP2GPDK180V0Env'
        , )

register( id          = 'op2-gpdk180-v1'
        , entry_point = 'gace.envs:OP2GPDK180V1Env'
        , )

register( id          = 'op2-gpdk180-v3'
        , entry_point = 'gace.envs:OP2GPDK180V3Env'
        , )

## AC²E: OP3 - Un-Symmetrical Amplifier
register( id          = 'op3-xh035-v0'
        , entry_point = 'gace.envs:OP3XH035V0Env'
        , )

register( id          = 'op3-xh035-v1'
        , entry_point = 'gace.envs:OP3XH035V1Env'
        , )

register( id          = 'op3-xh035-v2'
        , entry_point = 'gace.envs:OP3XH035V2Env'
        , )

register( id          = 'op3-xh035-v3'
        , entry_point = 'gace.envs:OP3XH035V3Env'
        , )

#register( id          = 'op3-xh018-v0'
#        , entry_point = 'gace.envs:OP3XH018V0Env'
#        , )
#
#register( id          = 'op3-xh018-v1'
#        , entry_point = 'gace.envs:OP3XH018V1Env'
#        , )
#
#register( id          = 'op3-xh018-v3'
#        , entry_point = 'gace.envs:OP3XH018V3Env'
#        , )
#
#register( id          = 'op3-xt018-v0'
#        , entry_point = 'gace.envs:OP3XT018V0Env'
#        , )
#
#register( id          = 'op3-xt018-v1'
#        , entry_point = 'gace.envs:OP3XT018V1Env'
#        , )

#register( id          = 'op3-xt018-v3'
#        , entry_point = 'gace.envs:OP3XT018V3Env'
#        , )

register( id          = 'op3-sky130-v0'
        , entry_point = 'gace.envs:OP3SKY130V0Env'
        , )

register( id          = 'op3-sky130-v1'
        , entry_point = 'gace.envs:OP3SKY130V1Env'
        , )

register( id          = 'op3-sky130-v3'
        , entry_point = 'gace.envs:OP3SKY130V3Env'
        , )

register( id          = 'op3-gpdk180-v0'
        , entry_point = 'gace.envs:OP3GPDK180V0Env'
        , )

register( id          = 'op3-gpdk180-v1'
        , entry_point = 'gace.envs:OP3GPDK180V1Env'
        , )

register( id          = 'op3-gpdk180-v3'
        , entry_point = 'gace.envs:OP3GPDK180V3Env'
        , )

## AC²E: OP4 - Symmetrical Cascode Amplifier
register( id          = 'op4-xh035-v0'
        , entry_point = 'gace.envs:OP4XH035V0Env'
        , )

register( id          = 'op4-xh035-v1'
        , entry_point = 'gace.envs:OP4XH035V1Env'
        , )

register( id          = 'op4-xh035-v2'
        , entry_point = 'gace.envs:OP4XH035V2Env'
        , )

register( id          = 'op4-xh035-v3'
        , entry_point = 'gace.envs:OP4XH035V3Env'
        , )

#register( id          = 'op4-xh018-v0'
#        , entry_point = 'gace.envs:OP4XH018V0Env'
#        , )
#
#register( id          = 'op4-xh018-v1'
#        , entry_point = 'gace.envs:OP4XH018V1Env'
#        , )
#
#register( id          = 'op4-xh018-v3'
#        , entry_point = 'gace.envs:OP4XH018V3Env'
#        , )
#
#register( id          = 'op4-xt018-v0'
#        , entry_point = 'gace.envs:OP4XT018V0Env'
#        , )
#
#register( id          = 'op4-xt018-v1'
#        , entry_point = 'gace.envs:OP4XT018V1Env'
#        , )

#register( id          = 'op4-xt018-v3'
#        , entry_point = 'gace.envs:OP4XT018V3Env'
#        , )

register( id          = 'op4-sky130-v0'
        , entry_point = 'gace.envs:OP4SKY130V0Env'
        , )

register( id          = 'op4-sky130-v1'
        , entry_point = 'gace.envs:OP4SKY130V1Env'
        , )

register( id          = 'op4-sky130-v3'
        , entry_point = 'gace.envs:OP4SKY130V3Env'
        , )

register( id          = 'op4-gpdk180-v0'
        , entry_point = 'gace.envs:OP4GPDK180V0Env'
        , )

register( id          = 'op4-gpdk180-v1'
        , entry_point = 'gace.envs:OP4GPDK180V1Env'
        , )

register( id          = 'op4-gpdk180-v3'
        , entry_point = 'gace.envs:OP4GPDK180V3Env'
        , )

## AC²E: OP5 - Un-Symmetrical Cascode Amplifier
register( id          = 'op5-xh035-v0'
        , entry_point = 'gace.envs:OP5XH035V0Env'
        , )

register( id          = 'op5-xh035-v1'
        , entry_point = 'gace.envs:OP5XH035V1Env'
        , )

register( id          = 'op5-xh035-v2'
        , entry_point = 'gace.envs:OP5XH035V2Env'
        , )

register( id          = 'op5-xh035-v3'
        , entry_point = 'gace.envs:OP5XH035V3Env'
        , )

#register( id          = 'op5-xh018-v0'
#        , entry_point = 'gace.envs:OP5XH018V0Env'
#        , )
#
#register( id          = 'op5-xh018-v1'
#        , entry_point = 'gace.envs:OP5XH018V1Env'
#        , )
#
#register( id          = 'op5-xh018-v3'
#        , entry_point = 'gace.envs:OP5XH018V3Env'
#        , )
#
#register( id          = 'op5-xt018-v0'
#        , entry_point = 'gace.envs:OP5XT018V0Env'
#        , )
#
#register( id          = 'op5-xt018-v1'
#        , entry_point = 'gace.envs:OP5XT018V1Env'
#        , )

#register( id          = 'op5-xt018-v3'
#        , entry_point = 'gace.envs:OP5XT018V3Env'
#        , )

register( id          = 'op5-sky130-v0'
        , entry_point = 'gace.envs:OP5SKY130V0Env'
        , )

register( id          = 'op5-sky130-v1'
        , entry_point = 'gace.envs:OP5SKY130V1Env'
        , )

register( id          = 'op5-sky130-v3'
        , entry_point = 'gace.envs:OP5SKY130V3Env'
        , )

register( id          = 'op5-gpdk180-v0'
        , entry_point = 'gace.envs:OP5GPDK180V0Env'
        , )

register( id          = 'op5-gpdk180-v1'
        , entry_point = 'gace.envs:OP5GPDK180V1Env'
        , )

register( id          = 'op5-gpdk180-v3'
        , entry_point = 'gace.envs:OP5GPDK180V3Env'
        , )

## AC²E: OP6 - Miller Amplifier w/o passives
register( id          = 'op6-xh035-v0'
        , entry_point = 'gace.envs:OP6XH035V0Env'
        , )

register( id          = 'op6-xh035-v1'
        , entry_point = 'gace.envs:OP6XH035V1Env'
        , )

register( id          = 'op6-xh035-v2'
        , entry_point = 'gace.envs:OP6XH035V2Env'
        , )

register( id          = 'op6-xh035-v3'
        , entry_point = 'gace.envs:OP6XH035V3Env'
        , )

#register( id          = 'op6-xh018-v0'
#        , entry_point = 'gace.envs:OP6XH018V0Env'
#        , )
#
#register( id          = 'op6-xh018-v1'
#        , entry_point = 'gace.envs:OP6XH018V1Env'
#        , )
#
#register( id          = 'op6-xh018-v3'
#        , entry_point = 'gace.envs:OP6XH018V3Env'
#        , )
#
#register( id          = 'op6-xt018-v0'
#        , entry_point = 'gace.envs:OP6XT018V0Env'
#        , )
#
#register( id          = 'op6-xt018-v1'
#        , entry_point = 'gace.envs:OP6XT018V1Env'
#        , )

#register( id          = 'op6-xt018-v3'
#        , entry_point = 'gace.envs:OP6XT018V3Env'
#        , )

register( id          = 'op6-sky130-v0'
        , entry_point = 'gace.envs:OP6SKY130V0Env'
        , )

register( id          = 'op6-sky130-v1'
        , entry_point = 'gace.envs:OP6SKY130V1Env'
        , )

register( id          = 'op6-sky130-v3'
        , entry_point = 'gace.envs:OP6SKY130V3Env'
        , )

register( id          = 'op6-gpdk180-v0'
        , entry_point = 'gace.envs:OP6GPDK180V0Env'
        , )

register( id          = 'op6-gpdk180-v1'
        , entry_point = 'gace.envs:OP6GPDK180V1Env'
        , )

register( id          = 'op6-gpdk180-v3'
        , entry_point = 'gace.envs:OP6GPDK180V3Env'
        , )

## AC²E: OP7
# ...

## AC²E: OP8 - Wideswing Current Mirror
register( id          = 'op8-xh035-v0'
        , entry_point = 'gace.envs:OP8XH035V0Env'
        , )

register( id          = 'op8-xh035-v1'
        , entry_point = 'gace.envs:OP8XH035V1Env'
        , )

register( id          = 'op8-xh035-v2'
        , entry_point = 'gace.envs:OP8XH035V2Env'
        , )

register( id          = 'op8-xh035-v3'
        , entry_point = 'gace.envs:OP8XH035V3Env'
        , )

#register( id          = 'op8-xh018-v0'
#        , entry_point = 'gace.envs:OP8XH018V0Env'
#        , )
#
#register( id          = 'op8-xh018-v1'
#        , entry_point = 'gace.envs:OP8XH018V1Env'
#        , )
#
#register( id          = 'op8-xh018-v3'
#        , entry_point = 'gace.envs:OP8XH018V3Env'
#        , )
#
#register( id          = 'op8-xt018-v0'
#        , entry_point = 'gace.envs:OP8XT018V0Env'
#        , )
#
#register( id          = 'op8-xt018-v1'
#        , entry_point = 'gace.envs:OP8XT018V1Env'
#        , )

#register( id          = 'op8-xt018-v3'
#        , entry_point = 'gace.envs:OP8XT018V3Env'
#        , )

register( id          = 'op8-gpdk180-v0'
        , entry_point = 'gace.envs:OP8GPDK180V0Env'
        , )

register( id          = 'op8-gpdk180-v1'
        , entry_point = 'gace.envs:OP8GPDK180V1Env'
        , )

register( id          = 'op8-gpdk180-v3'
        , entry_point = 'gace.envs:OP8GPDK180V3Env'
        , )

## AC²E: OP9 - Cascode Wideswing Current Mirror
register( id          = 'op9-xh035-v0'
        , entry_point = 'gace.envs:OP9XH035V0Env'
        , )

register( id          = 'op9-xh035-v1'
        , entry_point = 'gace.envs:OP9XH035V1Env'
        , )

register( id          = 'op9-xh035-v2'
        , entry_point = 'gace.envs:OP9XH035V2Env'
        , )

register( id          = 'op9-xh035-v3'
        , entry_point = 'gace.envs:OP9XH035V3Env'
        , )

#register( id          = 'op9-xh018-v0'
#        , entry_point = 'gace.envs:OP9XH018V0Env'
#        , )
#
#register( id          = 'op9-xh018-v1'
#        , entry_point = 'gace.envs:OP9XH018V1Env'
#        , )
#
#register( id          = 'op9-xt018-v0'
#        , entry_point = 'gace.envs:OP9XT018V0Env'
#        , )
#
#register( id          = 'op9-xt018-v1'
#        , entry_point = 'gace.envs:OP9XT018V1Env'
#        , )

register( id          = 'op9-gpdk180-v0'
        , entry_point = 'gace.envs:OP9GPDK180V0Env'
        , )

register( id          = 'op9-gpdk180-v1'
        , entry_point = 'gace.envs:OP9GPDK180V1Env'
        , )

## AC²E: OP10
# ...

## AC²E: NAND4 - 4 NAND Gate Inverter Chain
register( id          = 'nand4-xh035-v1'
        , entry_point = 'gace.envs:NAND4XH035V1Env'
        , )

register( id          = 'nand4-xh035-v3'
        , entry_point = 'gace.envs:NAND4XH035V3Env'
        , )

register( id          = 'nand4-xh018-v1'
        , entry_point = 'gace.envs:NAND4XH018V1Env'
        , )

register( id          = 'nand4-xh018-v3'
        , entry_point = 'gace.envs:NAND4XH018V3Env'
        , )

register( id          = 'nand4-xt018-v1'
        , entry_point = 'gace.envs:NAND4XT018V1Env'
        , )

register( id          = 'nand4-xt018-v3'
        , entry_point = 'gace.envs:NAND4XT018V3Env'
        , )

register( id          = 'nand4-sky130-v1'
        , entry_point = 'gace.envs:NAND4SKY130V1Env'
        , )

register( id          = 'nand4-sky130-v3'
        , entry_point = 'gace.envs:NAND4SKY130V3Env'
        , )

register( id          = 'nand4-gpdk180-v1'
        , entry_point = 'gace.envs:NAND4GPDK180V1Env'
        , )

register( id          = 'nand4-gpdk180-v3'
        , entry_point = 'gace.envs:NAND4GPDK180V3Env'
        , )

## AC²E: ST1 - Schmitt Trigger
register( id          = 'st1-xh035-v1'
        , entry_point = 'gace.envs:ST1XH035V1Env'
        , )

register( id          = 'st1-xh035-v3'
        , entry_point = 'gace.envs:ST1XH035V3Env'
        , )

register( id          = 'st1-xh018-v1'
        , entry_point = 'gace.envs:ST1XH018V1Env'
        , )

register( id          = 'st1-xh018-v3'
        , entry_point = 'gace.envs:ST1XH018V3Env'
        , )

register( id          = 'st1-xt018-v1'
        , entry_point = 'gace.envs:ST1XT018V1Env'
        , )

register( id          = 'st1-xt018-v3'
        , entry_point = 'gace.envs:ST1XT018V3Env'
        , )

register( id          = 'st1-sky130-v1'
        , entry_point = 'gace.envs:ST1SKY130V1Env'
        , )

register( id          = 'st1-sky130-v3'
        , entry_point = 'gace.envs:ST1SKY130V3Env'
        , )

register( id          = 'st1-gpdk180-v1'
        , entry_point = 'gace.envs:ST1GPDK180V1Env'
        , )

register( id          = 'st1-gpdk180-v3'
        , entry_point = 'gace.envs:ST1GPDK180V3Env'
        , )
