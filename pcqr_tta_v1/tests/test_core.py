import copy
import unittest

import torch

from pcqr_tta_v1.core import (AdaptConfig, OnlineAdapter, TransitionBuffer,
                             absolute_loss, find_lewm, response_loss, rollout_latent)
from pcqr_tta_v1.tests.fixtures import TinyJEPA


def fill(adapter, n=5):
    g = torch.Generator().manual_seed(444)
    adapter.buffer.begin(torch.randn(4, generator=g))
    for _ in range(n):
        adapter.buffer.append(torch.randn(10, generator=g), torch.randn(4, generator=g))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(77)
        self.model = TinyJEPA()
        self.z = torch.randn(1, 1, 4)
        self.q = torch.randn(8, 5, 10)

    def test_centering_allows_common_translation_per_horizon(self):
        p = torch.randn(8, 5, 4)
        shift = torch.randn(1, 5, 4)
        w = torch.ones(5)
        self.assertLess(float(response_loss(p + shift, p, w)), 1e-12)
        self.assertGreater(float(absolute_loss(p + shift, p, w)), 0.01)

    def test_relative_response_change_is_penalized(self):
        p = torch.randn(8, 5, 4)
        self.assertGreater(float(response_loss(1.1 * p, p, torch.ones(5))), 0.0)

    def test_single_query_rejected(self):
        p = torch.zeros(1, 5, 4)
        with self.assertRaises(ValueError):
            response_loss(p, p, torch.ones(5))

    def test_rollout_matches_independent_reference(self):
        pixels = torch.randn(2, 3, 1, 3, 2, 2)
        # Solver repeats the same current observation across candidates.
        pixels = pixels[:, :1].expand(2, 3, 1, 3, 2, 2).clone()
        actions = torch.randn(2, 3, 5, 10)
        for hs in (1, 2, 3):
            reference = self.model.rollout({'pixels': pixels}, actions, history_size=hs)['predicted_emb']
            z = self.model.encode({'pixels': pixels[:, 0]})['emb']
            z = z[:, None].expand(2, 3, 1, 4).reshape(6, 1, 4)
            actual = rollout_latent(self.model, z, actions.reshape(6, 5, 10), hs).reshape(2, 3, 5, 4)
            torch.testing.assert_close(actual, reference[:, :, 1:])

    def test_rollout_preserves_autoregressive_gradients(self):
        actions = self.q[:1].clone().requires_grad_()
        out = rollout_latent(self.model, self.z, actions)
        out[:, -1].square().sum().backward()
        self.assertGreater(float(actions.grad[:, 0].abs().sum()), 0)

    def test_one_step_response_exactly_matches_obs(self):
        adapters = []
        for mode in ('obs', 'response', 'absolute', 'l2'):
            a = OnlineAdapter(copy.deepcopy(self.model), AdaptConfig(mode=mode, steps=1, strength=100))
            fill(a)
            result = a.adapt(self.z, self.q)
            self.assertEqual(result['status'], 'updated')
            self.assertEqual(result['steps'][0]['regularizer'], 0.0)
            adapters.append(a)
        for a in adapters[1:]:
            for (n, p), (m, other) in zip(a.named, adapters[0].named):
                self.assertEqual(n, m)
                torch.testing.assert_close(p, other, rtol=0, atol=0)

    def test_multiple_steps_response_has_nonzero_penalty(self):
        a = OnlineAdapter(self.model, AdaptConfig(steps=3, lr=1e-3))
        fill(a)
        r = a.adapt(self.z, self.q)
        self.assertEqual(r['steps'][0]['regularizer'], 0.0)
        self.assertGreater(r['steps'][1]['regularizer'], 0)
        self.assertEqual(r['applied_steps'], 3)

    def test_regularization_changes_update_after_first_step(self):
        obs = OnlineAdapter(copy.deepcopy(self.model), AdaptConfig(mode='obs', steps=3, lr=1e-3))
        rsp = OnlineAdapter(copy.deepcopy(self.model), AdaptConfig(mode='response', steps=3, lr=1e-3, strength=1e4))
        fill(obs); fill(rsp)
        obs.adapt(self.z, self.q); rsp.adapt(self.z, self.q)
        distance = sum(float((a - b).detach().abs().sum()) for (_, a), (_, b) in zip(obs.named, rsp.named))
        self.assertGreater(distance, 1e-6)

    def test_freezes_encoder_projector_buffers_and_dropout(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        fill(a)
        a.adapt(self.z, self.q)
        a.assert_frozen_unchanged()
        self.assertTrue(all(not m.training for m in self.model.modules()))
        self.assertTrue(all(not p.requires_grad for p in self.model.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.projector.parameters()))

    def test_reset_restores_parameters_buffers_and_adam(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        fill(a)
        a.adapt(self.z, self.q)
        self.assertTrue(a.optimizer.state)
        a.reset_episode()
        self.assertFalse(a.optimizer.state)
        self.assertEqual(len(a.buffer), 0)
        self.assertFalse(a.buffer.states)
        for n, p in self.model.state_dict().items():
            torch.testing.assert_close(p.cpu(), a.source[n], rtol=0, atol=0)

    def test_backprop_inside_cem_inference_mode(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        fill(a)
        with torch.inference_mode():
            z, q = self.z.clone(), self.q.clone()
            self.assertTrue(z.is_inference())
            r = a.adapt(z, q)
        self.assertEqual(r['status'], 'updated')
        self.assertGreater(r['steps'][0]['grad_norm'], 0)

    def test_no_global_rng_consumption(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        fill(a)
        state = torch.get_rng_state().clone()
        a.adapt(self.z, self.q)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_frozen_and_warmup_are_noops(self):
        for mode, status in [('frozen', 'frozen'), ('response', 'warmup')]:
            a = OnlineAdapter(copy.deepcopy(self.model), AdaptConfig(mode=mode))
            before = copy.deepcopy(a.model.state_dict())
            r = a.adapt(self.z, self.q)
            self.assertEqual(r['status'], status)
            for n, p in a.model.state_dict().items():
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_nonfinite_loss_rolls_back(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        a.buffer.begin(torch.ones(4))
        a.buffer.append(torch.ones(10), torch.ones(4) * 1e30)
        before = copy.deepcopy(self.model.state_dict())
        r = a.adapt(self.z, self.q)
        self.assertEqual(r['status'], 'rolled_back')
        self.assertEqual(r['applied_steps'], 0)
        for n, p in self.model.state_dict().items():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        self.assertFalse(a.optimizer.state)

    def test_config_rejects_invalid_values(self):
        for kwargs in ({'steps': 0}, {'mode': 'not_a_mode'}, {'query_count': 1},
                       {'lr': float('nan')}, {'train_parts': ('encoder',)},
                       {'horizon_weights': (1, 2)}):
            with self.assertRaises(ValueError):
                AdaptConfig(**kwargs)

    def test_ambiguous_model_rejected(self):
        wrapper = torch.nn.ModuleDict({'student': self.model, 'teacher': copy.deepcopy(self.model)})
        with self.assertRaises(TypeError):
            find_lewm(wrapper)
        wrapper = torch.nn.ModuleDict({'model': self.model})
        self.assertIs(find_lewm(wrapper)[1], self.model)

    def test_query_shape_rejected(self):
        a = OnlineAdapter(self.model, AdaptConfig())
        fill(a)
        with self.assertRaises(ValueError):
            a.adapt(self.z.expand(2, -1, -1), self.q)


class BufferTests(unittest.TestCase):
    def test_only_observed_horizons_valid(self):
        b = TransitionBuffer(6, 10)
        b.begin(torch.zeros(4))
        b.append(torch.ones(10), torch.ones(4))
        data = b.batch(5, 16, torch.device('cpu'))
        self.assertEqual(data['valid'].tolist(), [[True, False, False, False, False]])
        torch.testing.assert_close(data['targets'][0, 0], torch.ones(4))
        b.append(2 * torch.ones(10), 2 * torch.ones(4))
        data = b.batch(5, 16, torch.device('cpu'))
        self.assertEqual(data['valid'].sum().item(), 3)
        torch.testing.assert_close(data['z0'][:, 0, 0], torch.tensor([0., 1.]))

    def test_capacity_preserves_action_endpoint_alignment(self):
        b = TransitionBuffer(3, 10)
        b.begin(torch.zeros(4))
        for i in range(1, 8):
            b.append(torch.full((10,), float(i)), torch.full((4,), float(i)))
        self.assertEqual(b.total_completed, 7)
        self.assertEqual(len(b), 3)
        data = b.batch(3, 8, torch.device('cpu'))
        torch.testing.assert_close(data['z0'][:, 0, 0], torch.tensor([4., 5., 6.]))
        torch.testing.assert_close(data['actions'][:, 0, 0], torch.tensor([5., 6., 7.]))
        torch.testing.assert_close(data['targets'][:, 0, 0], torch.tensor([5., 6., 7.]))

    def test_reset_and_input_checks(self):
        b = TransitionBuffer(3, 10)
        with self.assertRaises(RuntimeError):
            b.append(torch.zeros(10), torch.zeros(4))
        b.begin(torch.zeros(4))
        with self.assertRaises(RuntimeError):
            b.begin(torch.zeros(4))
        with self.assertRaises(ValueError):
            b.append(torch.zeros(9), torch.zeros(4))
        b.clear()
        b.begin(torch.ones(4))
        self.assertEqual(len(b.states), 1)
        self.assertEqual(len(b), 0)


if __name__ == '__main__':
    unittest.main()
