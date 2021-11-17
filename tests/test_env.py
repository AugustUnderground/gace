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

    assert env.observation_space.shape[0] == obs.shape[0], \
            f'The dimensions of the observation space do not match.'

    #assert env.observation_space.contains(obs), \
    #       f'The observation returned by `reset` does not match the given space.'

    act = env.action_space.sample()

    assert env.action_space.shape[0] == act.shape[0], \
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

def _test_amp_v0(env_id):
    env = gym.make( env_id
                  , data_log_path   = ""
                  , random_target   = True
                  , )
    _test_env(env)
    env.close()

def _test_amp_v1(env_id):
    env = gym.make( env_id
                  , data_log_path   = ""
                  , random_target   = True
                  , )
    _test_env(env)
    env.close()

#def test_amps_xh035():
#    amps = [f'op{op}' for op in [1,2,3,4,5,6,8,9]]
#    for a in amps:
#        _test_amp_v0(f'gace:{a}-xh035-v0')
#        _test_amp_v1(f'gace:{a}-xh035-v1')

#def test_amps_sky130():
#    amps = [f'op{op}' for op in [2,3,4,5]]
#    for a in amps:
#        _test_amp_v0(f'gace:{a}-sky130-v0')
#        _test_amp_v1(f'gace:{a}-sky130-v1')

def test_amps_gpdk180():
    amps = [f'op{op}' for op in [1,2,3,4,5,6,8,9]]
    for a in amps:
        _test_amp_v0(f'gace:{a}-gpdk180-v0')
        _test_amp_v1(f'gace:{a}-gpdk180-v1')

#def _test_inv_v1(env_id):
#    env = gym.make(env_id, data_log_path = "")
#    _test_env(env)
#    env.close()
#
#def test_invs():
#    invs = ['nand4']
#    for i in invs:
#        _test_inv_v1(f'gace:{i}-xh035-v1')
#        _test_inv_v1(f'gace:{i}-sky130-v1')
#        _test_inv_v1(f'gace:{i}-gpdk180-v1')
#
#def _test_trg_v1(env_id):
#    env = gym.make(env_id, data_log_path = "")
#    _test_env(env)
#    env.close()
#
#def test_trgs():
#    trgs = [f'st{st}' for st in [1]]
#    for t in trgs:
#        _test_trg_v1(f'gace:{t}-xh035-v1')
#        _test_trg_v1(f'gace:{t}-sky130-v1')
#        _test_trg_v1(f'gace:{t}-gpdk180-v1')
