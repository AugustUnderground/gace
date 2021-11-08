import os
import gym
import numpy as np

HOME = os.path.expanduser('~')

def _test_env(env):
    assert isinstance(env, gym.Env), \
           f'The env must inherit from gym.Env.'

    assert hasattr(env, 'observation_space'), \
           f'The env must specify an observation_space attribute.'
    assert hasattr(env, 'action_space'), \
           f'The env must specify an action_space attribute.'

    assert isinstance(env.observation_space, gym.spaces.Space), \
           f'The observation_space must inherit from gym.spaces.Space.'
    assert isinstance(env.action_space, gym.spaces.Space), \
           f'The action_space must inherit from gym.spaces.Space.'

    if isinstance(env.action_space, gym.spaces.Dict):
        for s in env.action_space.values():
            assert isinstance(s, gym.spaces.Space), \
                  f'All spaces within a dict action space must inherit from gym.spaces.Space'

    assert len(env.observation_space.shape) in [1, 3], \
           f'The shape of the observation space should be ∈ [1,3].'

    assert np.any(np.abs(env.action_space.low) == np.abs(env.action_space.high)), \
           f'The action space should be symmetric.'

    assert (np.any(np.abs(env.action_space.low) <= 1) or \
            np.any(np.abs(env.action_space.high) <= 1)), \
           f'The action space is not ∈ [-1;1]'

    assert env.action_space.is_bounded(), \
           f'The action space should be bounded.' 

    obs = env.reset()

    assert isinstance(obs, np.ndarray), \
           f'The observation returned by `reset` must be a numpy array.'

    #assert env.observation_space.contains(obs), \
    #       f'The observation returned by `reset` does not match the given space.'

    act = env.action_space.sample()
    res = env.step(act)

    assert len(res) == 4, \
           f'The `step` function must return a tuple with four elements.' 

    o,r,d,i = res

    assert isinstance(o, np.ndarray), \
           f'The observation returned by `step` must be a numpy array.' 
    
    #assert env.observation_space.contains(o), \
    #       f'The observation returned by `step` does not lie within the space.'

    assert isinstance(r, (float, int)), \
           f'The reward returned by `step` must be of type float.'

    assert isinstance(d, bool), \
           f'The done signal returned by `step` must be of type bool.'

    assert isinstance(i, dict), \
           f'The info returned by `step` must be of type dict'

    modes = env.metadata.get('render.modes')

    assert modes != None, \
           f'The env does not specify any render mode.'

    assert 'human' in modes, \
           f'The env does not specify a human render mode.'

    for m in modes:
        env.render(mode=m)

    for _ in range(10):
        act = env.action_space.sample()
        obs, rew, don, inf = env.step(act)

        assert not np.any(np.isnan(obs)), \
               f'Observations contain NaNs.'
        assert not np.any(np.isinf(obs)), \
               f'Observations contain +/-Infs.'

def test_v0_xh035():
    amps = ['1', '2', '3', '4', '6']
    nmos_path = '/mnt/data/share/xh035-nmos-20211022-091316'
    pmos_path = '/mnt/data/share/xh035-pmos-20211022-084243'
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    ace_path  = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3'

    for a in amps:
        op = f'op{a}'
        op_path = f'{ace_path}/{op}'
        env = gym.make( f'gace:{op}-xh035-v0'
                      , pdk_path        = pdk_path
                      , ckt_path        = op_path
                      , nmos_path       = nmos_path
                      , pmos_path       = pmos_path
                      , data_log_prefix = None
                      , random_target   = False
                      , )
        _test_env(env)
        env.close()

def test_v1_xh035():
    amps = ['1', '2', '3', '4', '6']
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    ace_path  = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3'

    for a in amps:
        op = f'op{a}'
        op_path = f'{ace_path}/{op}'
        env = gym.make( f'gace:{op}-xh035-v1'
                      , pdk_path        = pdk_path
                      , ckt_path        = op_path
                      , data_log_prefix = None
                      , random_target   = False
                      , )
        _test_env(env)
        env.close()

def test_inv():
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    nand_path = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3/nand4'
    env = gym.make( f'gace:nand4-xh035-v1'
                  , pdk_path        = pdk_path
                  , ckt_path        = nand_path
                  , data_log_prefix = None
                  , random_start    = True
                  , )
    _test_env(env)
    env.close()

def test_trg():
    pdk_path = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    trg_path = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3/st1'
    env = gym.make( f'gace:st1-xh035-v1'
                  , pdk_path        = pdk_path
                  , ckt_path        = trg_path
                  , data_log_prefix = None
                  , random_start    = True
                  , )
    _test_env(env)
    env.close()
