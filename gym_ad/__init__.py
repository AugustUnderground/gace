from gym.envs.registration import register

## AC²E: OP1
register( id          = 'op1-xh035-v0'
        , entry_point = 'gym_ad.envs:MillerAmpXH035Env'
        , )

## AC²E: OP2
register( id          = 'op2-xh035-v0'
        , entry_point = 'gym_ad.envs:SymAmpXH035Env'
        , )

register( id          = 'op2-xh035-v1'
        , entry_point = 'gym_ad.envs:SymAmpXH035GeomEnv'
        , )

## AC²E: OP3
register( id          = 'op3-xh035-v0'
        , entry_point = 'gym_ad.envs:UnSymAmpXH035Env'
        , )

## AC²E: OP4
register( id          = 'op4-xh035-v0'
        , entry_point = 'gym_ad.envs:SymCasAmpXH035Env'
        , )

## AC²E: OP6
register( id          = 'op6-xh035-v0'
        , entry_point = 'gym_ad.envs:MillerAmpModXH035Env'
        , )
