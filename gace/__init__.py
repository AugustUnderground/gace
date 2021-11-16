# Function for checking custom environments.
from gace.util import check_env

## Environment Variants:
# 
# | Variant | Description                                |
# |---------+--------------------------------------------|
# | v0      | V0trical Action Space (gmoverid and fug)   |
# | v1      | V1etrical Action Space (W, L and M)        |
# | v2      | TBA                                        |
#
# Technolgies:
#   - X-Fab 350nm: xh035-3V3
#   - SkyWater 130nm: sky130-1V8
#   - TODO GPDK 180nm: gpdk180-1V2
#   - TODO PTM 130nm: ptm130-1V2

from gym.envs.registration import register

## AC²E: OP1 - Miller Amplifier
register( id          = 'op1-xh035-v0'
        , entry_point = 'gace.envs:OP1XH035V0Env'
        , )

register( id          = 'op1-xh035-v1'
        , entry_point = 'gace.envs:OP1XH035V1Env'
        , )

## AC²E: OP2 - Symmetrical Amplifier
register( id          = 'op2-xh035-v0'
        , entry_point = 'gace.envs:OP2XH035V0Env'
        , )

register( id          = 'op2-xh035-v1'
        , entry_point = 'gace.envs:OP2XH035V1Env'
        , )

register( id          = 'op2-sky130-v0'
        , entry_point = 'gace.envs:OP2SKY130V1Env'
        , )

register( id          = 'op2-sky130-v1'
        , entry_point = 'gace.envs:OP2SKY130V1Env'
        , )

## AC²E: OP3 - Un-Symmetrical Amplifier
register( id          = 'op3-xh035-v0'
        , entry_point = 'gace.envs:OP3XH035V0Env'
        , )

register( id          = 'op3-xh035-v1'
        , entry_point = 'gace.envs:OP3XH035V1Env'
        , )

## AC²E: OP4 - Symmetrical Cascode Amplifier
register( id          = 'op4-xh035-v0'
        , entry_point = 'gace.envs:OP4XH035V0Env'
        , )

register( id          = 'op4-xh035-v1'
        , entry_point = 'gace.envs:OP4XH035V1Env'
        , )

## AC²E: OP5 - Un-Symmetrical Cascode Amplifier
register( id          = 'op5-xh035-v0'
        , entry_point = 'gace.envs:OP5XH035V0Env'
        , )

register( id          = 'op5-xh035-v1'
        , entry_point = 'gace.envs:OP5XH035V1Env'
        , )

## AC²E: OP6 - Miller Amplifier w/o passives
register( id          = 'op6-xh035-v0'
        , entry_point = 'gace.envs:OP6XH035V0Env'
        , )

register( id          = 'op6-xh035-v1'
        , entry_point = 'gace.envs:OP6XH035V1Env'
        , )

## AC²E: NAND4 - 4 NAND Gate Inverter Chain
register( id          = 'nand4-xh035-v1'
        , entry_point = 'gace.envs:NAND4XH035V1Env'
        , )

register( id          = 'nand4-sky130-v1'
        , entry_point = 'gace.envs:NAND4SKY130V1Env'
        , )

register( id          = 'nand4-gpdk180-v1'
        , entry_point = 'gace.envs:NAND4GPDK180V1Env'
        , )

## AC²E: ST1 - Schmitt Trigger
register( id          = 'st1-xh035-v1'
        , entry_point = 'gace.envs:ST1XH035V1Env'
        , )

register( id          = 'st1-sky130-v1'
        , entry_point = 'gace.envs:ST1SKY130V1Env'
        , )

register( id          = 'st1-gpdk180-v1'
        , entry_point = 'gace.envs:ST1GPDK180V1Env'
        , )
