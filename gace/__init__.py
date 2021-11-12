# Function for checking custom environments.
from gace.util import check_env

## Environment Variants:
# 
# | Variant | Description                                |
# |---------+--------------------------------------------|
# | v0      | Electrical Action Space (gmoverid and fug) |
# | v1      | Geometrical Action Space (W, L and M)      |
# | v2      | TBA                                        |
#
# Technolgies:
#   - X-Fab XH035: 350nm
#   - TODO SkyWater130: 130nm
#   - TODO GPDK: [90nm, 180nm]
#   - TODO PTM: [90nm, 130nm]

from gym.envs.registration import register

## AC²E: OP1
register( id          = 'op1-xh035-v0'
        , entry_point = 'gace.envs:OP1XH035ElecEnv'
        , )

register( id          = 'op1-xh035-v1'
        , entry_point = 'gace.envs:OP1XH035GeomEnv'
        , )

## AC²E: OP2
register( id          = 'op2-xh035-v0'
        , entry_point = 'gace.envs:OP2XH035ElecEnv'
        , )

register( id          = 'op2-xh035-v1'
        , entry_point = 'gace.envs:OP2XH035GeomEnv'
        , )

register( id          = 'op2-sky130-v1'
        , entry_point = 'gace.envs:OP2SKY130GeomEnv'
        , )

## AC²E: OP3
register( id          = 'op3-xh035-v0'
        , entry_point = 'gace.envs:OP3XH035ElecEnv'
        , )

register( id          = 'op3-xh035-v1'
        , entry_point = 'gace.envs:OP3XH035GeomEnv'
        , )

## AC²E: OP4
register( id          = 'op4-xh035-v0'
        , entry_point = 'gace.envs:OP4XH035ElecEnv'
        , )

register( id          = 'op4-xh035-v1'
        , entry_point = 'gace.envs:OP4XH035GeomEnv'
        , )

## AC²E: OP5
register( id          = 'op5-xh035-v0'
        , entry_point = 'gace.envs:OP5XH035ElecEnv'
        , )

register( id          = 'op5-xh035-v1'
        , entry_point = 'gace.envs:OP5XH035GeomEnv'
        , )

## AC²E: OP6
register( id          = 'op6-xh035-v0'
        , entry_point = 'gace.envs:OP6XH035ElecEnv'
        , )

register( id          = 'op6-xh035-v1'
        , entry_point = 'gace.envs:OP6XH035GeomEnv'
        , )

## AC²E: NAND4
register( id          = 'nand4-xh035-v1'
        , entry_point = 'gace.envs:NAND4XH035GeomEnv'
        , )

## AC²E: ST1
register( id          = 'st1-xh035-v1'
        , entry_point = 'gace.envs:ST1XH035GeomEnv'
        , )
