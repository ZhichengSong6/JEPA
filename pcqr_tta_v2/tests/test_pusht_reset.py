"""CPU tests of the isolated PushT physical reset contract.

These tests use fake bodies; they do NOT prove real PyMunk determinism.
"""
import unittest
from types import SimpleNamespace

from pcqr_tta_v2.pusht_reset import (
    RESET_MODES, fresh_set_state, physics_snapshot, physics_hidden_sha256,
    scoped_pusht_reset,
)


class Body:
    def __init__(self):
        self.position = (0.0, 0.0)
        self.velocity = (10.0, 20.0)
        self.angular_velocity = 1.0
        self.angle = 0.0
        self.force = (1.0, 2.0)
        self.torque = 3.0
        self.activations = 0

    def activate(self):
        self.activations += 1


class Space:
    def __init__(self):
        self.damping = 1.
        self.iterations = 10
        self.reindexed = []
        self.step_calls = 0

    def reindex_shapes_for_body(self, body):
        self.reindexed.append(body)

    def step(self, dt):
        self.step_calls += 1


class Environment:
    def __init__(self):
        self.agent = Body()
        self.block = Body()
        self.space = Space()
        self.dt = 0.01

    def _get_obs(self):
        return [*self.agent.position, *self.block.position,
                self.block.angle, *self.agent.velocity]


class PushT(Environment):
    __module__ = 'stable_worldmodel.envs.pusht.env'

    def _set_state(self, state):
        s = list(state)
        self.agent.position = tuple(s[:2])
        self.block.position = tuple(s[2:4])
        self.block.angle = s[4]
        self.agent.velocity = tuple(s[5:7])
        self.space.step(self.dt)


def fake_world():
    env = PushT()
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(
        envs=SimpleNamespace(unwrapped=SimpleNamespace(envs=[wrapped]))
    ), env


class ResetTests(unittest.TestCase):
    def test_fresh_dynamics_do_not_step_physics(self):
        env = PushT()
        state = [1., 2., 3., 4., 5., 6., 7.]
        fresh_set_state(env, state)
        self.assertEqual(env._get_obs(), state)
        snap = physics_snapshot(env)
        self.assertEqual(snap['block_velocity'], [0., 0.])
        self.assertEqual(snap['block_angular_velocity'], 0.)
        self.assertEqual(snap['block_force'], [0., 0.])
        self.assertEqual(snap['block_torque'], 0.)
        self.assertEqual(env.space.step_calls, 0)
        self.assertEqual(len(env.space.reindexed), 2)
        self.assertEqual(env.block.activations, 1)
        self.assertEqual(physics_hidden_sha256(snap), physics_hidden_sha256(physics_snapshot(env)))

    def test_bad_state_rejected_without_step(self):
        env = PushT()
        for values in ([1., 2.], [float('nan')] * 7):
            with self.assertRaises(ValueError):
                fresh_set_state(env, values)
        self.assertEqual(env.space.step_calls, 0)

    def test_legacy_mode_preserves_source(self):
        world, env = fake_world()
        with scoped_pusht_reset(world, 'legacy') as probe:
            env._set_state([1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(probe.patches_applied, 0)
        self.assertEqual(env.space.step_calls, 1)

    def test_scoped_patch_is_applied_and_restored(self):
        world, env = fake_world()
        original = env._set_state.__func__
        with scoped_pusht_reset(world, 'fresh_dynamics') as probe:
            env._set_state([1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(probe.patches_applied, 1)
            self.assertEqual(env.space.step_calls, 0)
            self.assertEqual(probe.snapshot()['block_velocity'], [0., 0.])
        self.assertIs(env._set_state.__func__, original)
        env._set_state([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(env.space.step_calls, 1)

    def test_method_restored_on_failure(self):
        world, env = fake_world()
        original = env._set_state.__func__
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with scoped_pusht_reset(world, 'fresh_dynamics'):
                raise RuntimeError('sentinel')
        self.assertIs(env._set_state.__func__, original)

    def test_invalid_mode_fails_closed(self):
        world, env = fake_world()
        with self.assertRaises(ValueError):
            with scoped_pusht_reset(world, 'unknown'):
                pass
        self.assertNotIn('_set_state', vars(env))

    def test_mode_names(self):
        self.assertEqual(RESET_MODES, ('legacy', 'fresh_dynamics'))


if __name__ == '__main__':
    unittest.main()
