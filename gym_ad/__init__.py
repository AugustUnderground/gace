from gym.envs.registration import register

register( id          = 'sym-amp-xh035-v0'
        , entry_point = 'gym_ad.envs:SymAmpXH035Env'
        , )

#register( id='miller-amp-xh035-v0'
#        , entry_point='cyrcus.envs:MillerAmpXH035Env'
#        , )
