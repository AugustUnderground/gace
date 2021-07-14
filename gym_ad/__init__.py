from gym.envs.registration import register

register( id          = 'symmetrical-amplifier-v0'
        , entry_point = 'gym_ad.envs:SymAmpEnv'
        , )

#register( id='miller-amplifier-v0'
#        , entry_point='cyrcus.envs:MillerAmpEnv'
#        , )
