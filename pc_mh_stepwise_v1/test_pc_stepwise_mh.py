#!/usr/bin/env python3
"""CPU algebra, gradient and integration checks; no checkpoints/data needed.

On the project server, --with-repo uses the real upstream rollout and sampler.
Without that flag the integration tests use small, explicit fixture helpers;
the nonlinear target/algebra tests always exercise the real new implementation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
WITH_REPO = "--with-repo" in sys.argv
if WITH_REPO:
    sys.argv.remove("--with-repo")
import planner_context_stepwise_mh as new
import train_pc_stepwise_mh as runner


def reference_rollout(model, current_emb, plan_actions, predictor_max_history=3):
    history = current_emb[:, None]
    result = []
    for k in range(plan_actions.shape[1]):
        act = model.action_encoder(plan_actions[:, :k+1])
        nxt = model.predict(history[:, -predictor_max_history:], act[:, -predictor_max_history:])[:, -1:]
        result.append(nxt)
        history = torch.cat([history, nxt], 1)
    return torch.cat(result, 1)


def fixture_helpers():
    ald = ModuleType("anchored_local_dynamics")
    pc = ModuleType("planner_context_mh")
    def repeat(x, count):
        return x[:, None].expand(x.shape[0], count, *x.shape[1:]).reshape(-1, *x.shape[1:])
    def mask(positions, horizon, dtype, device):
        valid = torch.arange(horizon, device=device)[None, :] >= positions[:, None]
        return valid.to(dtype)[None, :, :, None]
    def mse(error, m):
        per_cell = error.pow(2).mean(-1, keepdim=True)
        m = m.expand(error.shape[0], -1, -1, -1)
        return (per_cell * m).sum() / m.sum().clamp_min(1.0)
    def mean(x, m):
        return (x*m).sum() / m.expand(x.shape[0], -1, -1).sum().clamp_min(1.0)
    def encode(model, batch):
        model.encoder.eval(); model.projector.eval()
        with torch.no_grad():
            return model.encode({"pixels": batch["pixels"]})["emb"].detach()
    def sample(normalized_actions, action_mean, action_std, history_size, horizon,
               radius, directions_per_position):
        plus, minus, radii, positions = [], [], [], []
        for p in range(horizon):
            for _ in range(directions_per_position):
                v = torch.randn_like(normalized_actions[:, 0])
                v = radius * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                a, b = normalized_actions.clone(), normalized_actions.clone()
                a[:, history_size-1+p] += v; b[:, history_size-1+p] -= v
                plus.append(a); minus.append(b)
                radii.append(v.norm(dim=-1)); positions.append(p)
        return torch.stack(plus,1), torch.stack(minus,1), torch.stack(radii,1), torch.tensor(positions)
    ald._repeat_for_candidates = repeat
    ald._mh_causal_mask = mask
    ald._mh_masked_mse = mse
    ald._mh_masked_mean = mean
    ald._frozen_visual_encode = encode
    ald._sample_blockwise_probe_actions = sample
    pc.planner_context_rollout = reference_rollout
    return {ald.__name__: ald, pc.__name__: pc}


class ScalarTeacher(nn.Module):
    def __init__(self, nonlinear=True, bias=0.2):
        super().__init__()
        self.action_encoder = nn.Identity()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float64))
        self.nonlinear = nonlinear
        self.contexts = []
    def predict(self, emb, act):
        self.contexts.append((emb.detach().clone(), act.detach().clone()))
        z = emb.square() if self.nonlinear else 0.7 * emb
        return z + act + self.bias


class TinyAction(nn.Linear):
    def __init__(self, dim):
        super().__init__(dim, dim)
        self.patch_embed = NS(in_channels=dim)


class TinyModel(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.action_encoder = TinyAction(dim)
        self.predictor = nn.Linear(dim, dim)
        self.pred_proj = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))
    def encode(self, info):
        pixels = info["pixels"]
        b,t,d = pixels.shape
        z = self.projector(self.encoder(pixels.reshape(b*t,d))).reshape(b,t,d)
        info["emb"] = z
        return info
    def predict(self, emb, act):
        # Deliberately use older context as well as the last frame.
        raw = torch.tanh(self.predictor(emb) + act + 0.1*emb.mean(1, keepdim=True))
        b,t,d = raw.shape
        return self.pred_proj(raw.reshape(b*t,d)).reshape(b,t,d)


class Capture:
    def __init__(self, model, teacher):
        self.model, self.teacher_model = model, teacher
        self.logs = {}
    def log_dict(self, values, **kwargs):
        self.logs.update(values)


def config():
    return NS(wm=NS(history_size=3, num_preds=1), mh_ald=NS(
        rollout_horizon=5, perturb_radius=0.1, directions_per_position=1,
        probes_per_chunk=2, tf_weight=1., rollout_weight=1., weight=1.,
        planner_context=NS(enabled=True, observation_history=1, predictor_max_history=3)))


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
    def test_zero_probe_recovers_every_observation(self):
        t = ScalarTeacher()
        obs = torch.randn(2,6,1,dtype=torch.float64)*0.3
        u = torch.randn(2,5,1,dtype=torch.float64)*0.1
        y = new.stepwise_observation_targets(t,obs,u,u)
        torch.testing.assert_close(y,obs[:,1:],rtol=0,atol=1e-12)
    def test_horizon_one_equals_original_anchor(self):
        t = ScalarTeacher()
        obs = torch.randn(2,2,1,dtype=torch.float64)
        u = torch.zeros(2,1,1,dtype=torch.float64)
        up = u + .12
        expected = obs[:,1:] + reference_rollout(t,obs[:,0],up) - reference_rollout(t,obs[:,0],u)
        torch.testing.assert_close(new.stepwise_observation_targets(t,obs,u,up),expected)
    def test_nonlinear_bias_example(self):
        t = ScalarTeacher()
        obs = torch.tensor([[[0.],[1.],[1.]]],dtype=torch.float64)
        u = torch.tensor([[[1.],[0.]]],dtype=torch.float64)
        up = torch.tensor([[[1.1],[0.]]],dtype=torch.float64)
        y = new.stepwise_observation_targets(t,obs,u,up)
        torch.testing.assert_close(y,torch.tensor([[[1.1],[1.21]]],dtype=torch.float64))
        old = obs[:,1:] + reference_rollout(t,obs[:,0],up) - reference_rollout(t,obs[:,0],u)
        self.assertAlmostEqual(old[0,-1,0].item(),1.25)
        self.assertGreater((old-y).abs().max().item(),0.039)
    def test_affine_equivalence(self):
        t = ScalarTeacher(nonlinear=False)
        obs = torch.randn(2,6,1,dtype=torch.float64)
        u = torch.randn(2,5,1,dtype=torch.float64)*.1
        up = u + torch.randn_like(u)*.03
        new_target = new.stepwise_observation_targets(t,obs,u,up)
        old = obs[:,1:]+reference_rollout(t,obs[:,0],up)-reference_rollout(t,obs[:,0],u)
        torch.testing.assert_close(new_target,old,rtol=0,atol=1e-12)
    def test_context_warmup_and_no_future_input(self):
        t=ScalarTeacher()
        obs=torch.arange(6,dtype=torch.float64).reshape(1,6,1)*.1
        u=torch.arange(5,dtype=torch.float64).reshape(1,5,1)*.02
        ref=new.factual_teacher_predictions(t,obs,u)
        self.assertEqual([x.shape[1] for x,a in t.contexts],[1,2,3,3,3])
        for k,(x,a) in enumerate(t.contexts):
            left=max(0,k-2)
            torch.testing.assert_close(x,obs[:,left:k+1])
            torch.testing.assert_close(a,u[:,left:k+1])
        t.contexts=[]
        new.stepwise_observation_targets(t,obs,u,u+.01,reference_predictions=ref)
        self.assertEqual([x.shape[1] for x,a in t.contexts],[1,2,3,3,3])
    def test_late_action_is_causal(self):
        t=ScalarTeacher()
        obs=torch.randn(2,6,1,dtype=torch.float64)*.3
        u=torch.zeros(2,5,1,dtype=torch.float64)
        up=u.clone(); up[:,3]+=.1
        y=new.stepwise_observation_targets(t,obs,u,up)
        torch.testing.assert_close(y[:,:3],obs[:,1:4],rtol=0,atol=1e-12)
        self.assertGreater((y[:,3:]-obs[:,4:]).abs().max().item(),.09)
    def test_targets_are_detached(self):
        t=ScalarTeacher()
        obs=torch.randn(2,3,1,dtype=torch.float64,requires_grad=True)
        u=torch.zeros(2,2,1,dtype=torch.float64,requires_grad=True)
        y=new.stepwise_observation_targets(t,obs,u,u+.01)
        self.assertFalse(y.requires_grad); self.assertIsNone(y.grad_fn)
        self.assertTrue(all(p.grad is None for p in t.parameters()))
    def test_precomputed_reference_matches_direct(self):
        t=ScalarTeacher()
        obs=torch.randn(2,6,1,dtype=torch.float64)*.2
        u=torch.randn(2,5,1,dtype=torch.float64)*.1
        ref=new.factual_teacher_predictions(t,obs,u)
        a=new.stepwise_observation_targets(t,obs,u,u+.01)
        b=new.stepwise_observation_targets(t,obs,u,u+.01,reference_predictions=ref)
        torch.testing.assert_close(a,b)
    def test_bad_shapes_rejected(self):
        t=ScalarTeacher(); obs=torch.zeros(2,4,1); u=torch.zeros(2,2,1)
        with self.assertRaises(ValueError): new.stepwise_observation_targets(t,obs,u,u)
        with self.assertRaises(ValueError): new.stepwise_observation_targets(t,obs[:,:3],u,u,0)
    def test_no_input_mutation(self):
        t=ScalarTeacher(); obs=torch.randn(2,4,1,dtype=torch.float64)*.1
        u=torch.randn(2,3,1,dtype=torch.float64)*.1
        a,b=obs.clone(),u.clone()
        new.stepwise_observation_targets(t,obs,u,u+.02)
        torch.testing.assert_close(obs,a);torch.testing.assert_close(u,b)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not WITH_REPO:
            cls.fixture_patch=patch.dict(sys.modules,fixture_helpers());cls.fixture_patch.start()
        else:
            runner.verify_upstream()
            import anchored_local_dynamics, planner_context_mh  # noqa
    @classmethod
    def tearDownClass(cls):
        if not WITH_REPO: cls.fixture_patch.stop()
    def setUp(self):
        torch.manual_seed(11)
        self.student=TinyModel().eval()
        self.teacher=copy.deepcopy(self.student).eval().requires_grad_(False)
        self.student.encoder.requires_grad_(False)
        self.student.projector.requires_grad_(False)
        self.batch={"pixels":torch.randn(2,8,3)*.1,"action":torch.zeros(2,8,3)}
        self.cfg=config()
    def forward(self):
        c=Capture(self.student,self.teacher)
        out=new.pc_stepwise_mh_forward(c,copy.deepcopy(self.batch),"train",self.cfg,
                                       torch.zeros(1,3),torch.ones(1,3))
        return c,out
    def test_tf_and_demo_rollout_unchanged(self):
        from planner_context_mh import planner_context_rollout
        with torch.no_grad():
            z=self.student.encode({"pixels":self.batch["pixels"]})["emb"]
        pred=self.student.predict(z[:,:3],self.student.action_encoder(self.batch["action"][:,:3]))
        tf=(pred-z[:,1:4]).square().mean()
        r=planner_context_rollout(self.student,z[:,2],self.batch["action"][:,2:7],3)
        rollout=(r-z[:,3:8]).square().mean()
        c,o=self.forward()
        torch.testing.assert_close(o["tf_loss"],tf)
        torch.testing.assert_close(o["pc_stepwise_rollout_loss"],rollout)
    def test_gradient_destinations_and_buffers(self):
        sb={k:v.clone() for k,v in self.student.named_buffers()}
        ts={k:v.clone() for k,v in self.teacher.state_dict().items()}
        c,o=self.forward(); o["loss"].backward()
        for name in ("action_encoder","predictor","pred_proj"):
            grads=[p.grad for p in getattr(self.student,name).parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in self.student.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.student.projector.parameters()))
        for k,v in self.student.named_buffers():torch.testing.assert_close(v,sb[k],rtol=0,atol=0)
        for k,v in self.teacher.state_dict().items():torch.testing.assert_close(v,ts[k],rtol=0,atol=0)
    def test_midpoint_response_decomposition(self):
        c,o=self.forward()
        torch.testing.assert_close(o["pc_stepwise_mh_loss"],o["pc_stepwise_midpoint_loss"]+o["pc_stepwise_half_response_loss"])
        self.assertTrue(all(not v.requires_grad for v in c.logs.values()))
    def test_chunk_size_invariance_and_detach(self):
        with torch.no_grad():z=self.student.encode({"pixels":self.batch["pixels"]})["emb"][:,2:8]
        results=[]
        for chunk in (1,2,5):
            torch.manual_seed(123)
            results.append(new._paired_stepwise_predictions(
                self.student,self.teacher,z,self.batch["action"],torch.zeros(1,3),torch.ones(1,3),
                3,5,3,.1,1,chunk))
        for result in results[1:]:
            for i in range(7):torch.testing.assert_close(result[i],results[0][i],rtol=1e-5,atol=1e-6)
        self.assertTrue(results[0][0].requires_grad)
        self.assertFalse(results[0][2].requires_grad)
    def test_student_probe_does_not_receive_future_observations(self):
        with torch.no_grad():z=self.student.encode({"pixels":self.batch["pixels"]})["emb"][:,2:8]
        changed=z.clone();changed[:,1:]+=.2
        result=[]
        for obs in (z,changed):
            torch.manual_seed(12)
            result.append(new._paired_stepwise_predictions(
                self.student,self.teacher,obs,self.batch["action"],torch.zeros(1,3),torch.ones(1,3),
                3,5,3,.1,1,2))
        torch.testing.assert_close(result[0][0],result[1][0],rtol=0,atol=0)
        torch.testing.assert_close(result[0][1],result[1][1],rtol=0,atol=0)
        self.assertFalse(torch.allclose(result[0][2],result[1][2]))
    def test_bfloat16_forward_backward_finite(self):
        with torch.autocast("cpu",dtype=torch.bfloat16):c,o=self.forward()
        self.assertTrue(torch.isfinite(o["loss"]))
        o["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.student.parameters() if p.grad is not None))
    @unittest.skipUnless(WITH_REPO,"requires actual project source")
    def test_one_step_full_loss_and_gradients_match_old_code(self):
        import planner_context_mh as old
        self.cfg.mh_ald.rollout_horizon=1
        before=copy.deepcopy(self.student)
        old_capture=Capture(before,copy.deepcopy(self.teacher))
        torch.manual_seed(17)
        old_out=old.pc_mh_ald_forward(old_capture,copy.deepcopy(self.batch),"train",self.cfg,
                                      torch.zeros(1,3),torch.ones(1,3))
        torch.manual_seed(17); _,new_out=self.forward()
        torch.testing.assert_close(old_out["loss"],new_out["loss"],rtol=1e-5,atol=1e-6)
        old_out["loss"].backward();new_out["loss"].backward()
        for a,b in zip(before.parameters(),self.student.parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)


def base_config():
    OmegaConf=runner._omega()
    return OmegaConf.create({
        "subdir":"old_pc_mh","output_model_name":"old_model","seed":3072,
        "wandb":{"enabled":False,"config":{}},
        "trainer":{"devices":4,"num_nodes":1,"max_epochs":10,"sync_batchnorm":False,"precision":"bf16","strategy":"ddp"},
        "loader":{"batch_size":16,"num_workers":4},
        "data":{"dataset":{"num_steps":"${eval:'${wm.num_preds} + 7'}","keys_to_load":["pixels","action","proprio"]}},
        "wm":{"history_size":3,"num_preds":1},
        "mh_ald":{"enabled":True,"init_policy":"lewm_epoch_10","teacher_policy":"lewm_epoch_10","rollout_horizon":5,
                  "planner_context":{"enabled":True,"observation_history":1,"predictor_max_history":3}}
    })


class ConfigurationTests(unittest.TestCase):
    def test_formal_changes_only_run_identifiers(self):
        cfg,changed,ng=runner.derive_configuration(base_config(),"formal","v1")
        self.assertEqual({x["key"] for x in changed},{"subdir","output_model_name"})
        self.assertEqual(ng,4);self.assertEqual(cfg.loader.batch_size,16)
        self.assertEqual(cfg.trainer.max_epochs,10)
        self.assertEqual(cfg.mh_ald.init_policy,"lewm_epoch_10")
    def test_smoke_uses_same_parallelism(self):
        cfg,changed,ng=runner.derive_configuration(base_config(),"smoke","v1")
        self.assertEqual(cfg.trainer.max_steps,2);self.assertEqual(ng,4)
        self.assertEqual(cfg.loader.batch_size,16)
    def test_auto_gpu_count_rejected(self):
        c=base_config();c.trainer.devices="auto"
        with self.assertRaises(ValueError):runner.derive_configuration(c,"formal","v1")
    def test_different_parent_rejected(self):
        c=base_config();c.mh_ald.teacher_policy="other_epoch_10"
        with self.assertRaises(ValueError):runner.derive_configuration(c,"formal","v1")
    def test_output_escape_rejected(self):
        for v in ("../x","/tmp/x","", "."):
            with self.assertRaises(ValueError):runner.path_in_root(HERE,v)
    def test_unsafe_resolver_rejected(self):
        self.assertEqual(runner._arithmetic("3+5"),8)
        with self.assertRaises(ValueError):runner._arithmetic("__import__('os').getcwd()")


class RunnerFixtureTests(unittest.TestCase):
    def test_state_fingerprint_supports_scalar_and_bfloat16(self):
        state={"scalar":torch.tensor(1),"bf16":torch.ones(2,3,dtype=torch.bfloat16)}
        self.assertEqual(runner._state_digest(state),runner._state_digest(state))
        altered={**state,"scalar":torch.tensor(2)}
        self.assertNotEqual(runner._state_digest(state),runner._state_digest(altered))

    def test_prepare_run_export_and_no_resume_with_fixture_backend(self):
        # This is a runner/control-flow test, NOT a real Lightning/Slurm test.
        import argparse, contextlib, io, json, os, tempfile
        OmegaConf=runner._omega()
        backend=ModuleType("train_pc_mh_ald")
        torch.manual_seed(44)
        parent=TinyModel().eval()
        backend._load_policy_model=lambda policy: copy.deepcopy(parent)
        backend.pc_mh_ald_forward=lambda *a,**kw: None
        class Callback:
            def __init__(self,dirpath,filename,epoch_interval=1):
                self.dirpath=Path(dirpath);self.filename=filename;self.epoch_interval=epoch_interval
        backend.ModelObjectCallBack=Callback
        def train(cfg):
            student=backend._load_policy_model(cfg.mh_ald.init_policy)
            student.encoder.requires_grad_(False);student.projector.requires_grad_(False)
            teacher=backend._load_policy_model(cfg.mh_ald.teacher_policy).requires_grad_(False).eval()
            module=Capture(student,teacher)
            trainer=NS(world_size=1,is_global_zero=True,current_epoch=0,max_epochs=1,
                       global_step=0,callback_metrics={})
            callback=backend.ModelObjectCallBack(Path(os.environ["STABLEWM_HOME"])/cfg.subdir,
                                                cfg.output_model_name,1)
            callback.on_fit_start(trainer,module)
            batch={"pixels":torch.randn(2,8,3)*.1,"action":torch.zeros(2,8,3)}
            result=backend.pc_mh_ald_forward(module,batch,"train",config(),torch.zeros(1,3),torch.ones(1,3))
            result["loss"].backward()
            callback.on_after_backward(trainer,module)
            torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=.001).step()
            trainer.global_step=1;trainer.callback_metrics=module.logs
            callback.on_train_batch_end(trainer,module,result,batch,0)
            callback.on_train_epoch_end(trainer,module)
        backend.run=train
        with tempfile.TemporaryDirectory(prefix="_runner_test_",dir=HERE) as temporary:
            root=Path(temporary);base=root/"baseline";base.mkdir()
            c=base_config();c.trainer.devices=1
            OmegaConf.save(c,base/"config.yaml")
            env={"STABLEWM_HOME":str(root),"SLURM_JOB_ID":"synthetic-test-only",
                 "SLURM_NTASKS":"1","SLURM_PROCID":"0","RANK":"0","LOCAL_RANK":"0"}
            with patch.dict(os.environ,env), patch.object(runner,"verify_upstream",return_value={}), \
                 patch.dict(sys.modules,{**fixture_helpers(),"train_pc_mh_ald":backend}), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                runner.prepare(argparse.Namespace(base_config=str(base/"config.yaml"),mode="smoke",tag="test"))
                out=root/"old_pc_mh_stepwise_test_smoke"
                args=argparse.Namespace(execution=str(out/"execution.json"))
                runner.run_prepared(args)
                complete=json.loads((out/"training_complete.json").read_text())
                restored=torch.load(complete["final_checkpoint"],map_location="cpu",weights_only=False)
                self.assertIsInstance(restored,TinyModel)
                self.assertTrue((out/"fit_audit.json").exists())
                self.assertTrue((out/"training_progress.csv").exists())
                with self.assertRaises(FileExistsError):runner.run_prepared(args)


if __name__=="__main__":
    torch.set_num_threads(1)
    print("UPSTREAM INTEGRATION: " + ("real repository helpers" if WITH_REPO else "local fixture helpers"),flush=True)
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print(f"STEPWISE UNIT TESTS PASS: {result.testsRun} tests ({len(result.skipped)} skipped)")
    sys.exit(0 if result.wasSuccessful() else 1)
