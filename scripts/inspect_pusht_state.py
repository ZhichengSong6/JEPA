#!/usr/bin/env python3
"""Small sanity check for the ground-truth state/action semantics in PushT."""

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments


def main():
    env = gym.make('swm/PushT-v1', render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    state = np.asarray(obs['state'])

    print('state =', state)
    print('  pusher_xy       =', state[0:2])
    print('  block_xy        =', state[2:4])
    print('  block_theta_rad =', state[4])
    print('  pusher_velocity =', state[5:7])
    print('proprio =', np.asarray(obs['proprio']))
    print('action_space =', env.action_space)

    action = np.array([0.2, -0.1], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    print('\naction =', action)
    print('next_state =', np.asarray(next_obs['state']))
    print('n_contacts =', next_info.get('n_contacts'))
    print('reward =', reward, 'terminated =', terminated, 'truncated =', truncated)
    env.close()


if __name__ == '__main__':
    main()
