from gym.envs.registration import register

# AC²E: OP1
register( id          = 'miller-amp-xh035-v0'
        , entry_point = 'gym_ad.envs:MillerAmpXH035Env'
        , )

# AC²E: OP2
register( id          = 'sym-amp-xh035-v0'
        , entry_point = 'gym_ad.envs:SymAmpXH035Env'
        , )

# AC²E: OP3
register( id          = 'sym-amp-xh035-v1'
        , entry_point = 'gym_ad.envs:UnSymAmpXH035Env'
        , )

# AC²E: OP4
register( id          = 'symcas-amp-xh035-v0'
        , entry_point = 'gym_ad.envs:SymCasAmpXH035Env'
        , )

# AC²E: OP6
register( id          = 'miller-amp-xh035-v1'
        , entry_point = 'gym_ad.envs:MillerAmpModXH035Env'
        , )
