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

    assert evn.observation_space.shape[0] == obs.shape[0], \
            f'The dimensions of the observation space do not match.'

    #assert env.observation_space.contains(obs), \
    #       f'The observation returned by `reset` does not match the given space.'

    act = env.action_space.sample()

    assert evn.action_space.shape[0] == act.shape[0], \
            f'The dimensions of the action space do not match.'

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

def _test_amp_v0(env_id, pdk_path, amp_path, nmos_path, pmos_path):
    env = gym.make( env_id
                  , pdk_path        = pdk_path
                  , ckt_path        = amp_path
                  , nmos_path       = nmos_path
                  , pmos_path       = pmos_path
                  , data_log_prefix = ""
                  , random_target   = True
                  , )
    _test_env(env)
    env.close()

def _test_amp_v1(env_id, pdk_path, amp_path):
    env = gym.make( env_id
                  , pdk_path        = pdk_path
                  , ckt_path        = amp_path
                  , data_log_prefix = ""
                  , random_target   = True
                  , )
    _test_env(env)
    env.close()

def _test_amp_xh035(a):
    nmos_path = '/mnt/data/share/xh035-nmos-20211022-091316'
    pmos_path = '/mnt/data/share/xh035-pmos-20211022-084243'
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    ace_path  = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3'

    amp_path = f'{ace_path}/{a}'

    _test_amp_v0(f'gace:{a}-xh035-v0', pdk_path, amp_path, nmos_path, pmos_path)
    _test_amp_v1(f'gace:{a}-xh035-v1', pdk_path, amp_path)

def test_amp():
    amps = [f'op{(op+1)}' for op in range(6)]
    for a in amps:
        _test_amp_xh035(a)

def _test_inv_v1(env_id, pdk_path, nand_path):
    env = gym.make( env_id
                  , pdk_path        = pdk_path
                  , ckt_path        = nand_path
                  , data_log_prefix = ""
                  , random_start    = True
                  , )
    _test_env(env)
    env.close()

def _test_inv_xh035(i):
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    ace_path  = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3'

    inv_path = f'{ace_path}/{i}'

    _test_inv_v1(f'gace:{i}-xh035-v1', pdk_path, inv_path)

def test_inv():
    invs = ['nand4']
    for i in invs:
        _test_inv_xh035(i)

def _test_trg_v1(env_id, pdk_path, trg_path):
    env = gym.make( env_id
                  , pdk_path        = pdk_path
                  , ckt_path        = trg_path
                  , data_log_prefix = ""
                  , random_start    = True
                  , )
    _test_env(env)
    env.close()

def _test_trg_xh035(t):
    pdk_path  = '/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos'
    ace_path  = f'{HOME}/Workspace/ACE/ace/resource/xh035-3V3'

    trg_path = f'{ace_path}/{t}'

    _test_trg_v1(f'gace:{t}-xh035-v1', pdk_path, trg_path)

def test_trg():
    trgs = [f'st{(st+1)}' for st in range(1)]
    for t in trgs:
        _test_trg_xh035(t)
