import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
    from small_agent.training.quality import HealthWindow, LayerDiagnostics, domain_plan, enqueue_checkpoint, preserve_runtime
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "run in the frozen remote Python environment")
class TrainingQualityTests(unittest.TestCase):
    def test_every_update_clip_fraction_not_last_step(self):
        window = HealthWindow(torch.device("cpu"), 1.)
        for norm in (2., 3., .5, .1):
            window.add(torch.tensor(20.), 10, torch.tensor(norm), .2)
        report = window.report()
        self.assertEqual(report["clip_fraction_window"], .5)
        self.assertAlmostEqual(report["grad_norm_window_mean"], 1.4)
        self.assertEqual(report["grad_norm_window_max"], 3.)
        self.assertEqual(report["predicted_tokens_window"], 40)

    def test_loss_is_target_weighted_without_changing_objective(self):
        window = HealthWindow(torch.device("cpu"), 1.)
        window.add(torch.tensor(2. * 10), 10, torch.tensor(.5), .1)
        window.add(torch.tensor(5. * 2), 2, torch.tensor(.5), .1)
        self.assertAlmostEqual(window.report()["loss_token_weighted"], 2.5)
        window.reset()
        with self.assertRaises(ValueError): window.report()

    def test_nonfinite_statistics_rejected(self):
        window = HealthWindow(torch.device("cpu"), 1.)
        window.add(torch.tensor(float("nan")), 1, torch.tensor(1.), .1)
        with self.assertRaises(FloatingPointError): window.report()

    def test_eval_restores_rng_generator_and_module_modes_even_on_failure(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.3))
        model.train(); model[1].eval()
        generator = torch.Generator().manual_seed(81)
        cpu, loader, python = torch.get_rng_state(), generator.get_state(), random.getstate()
        with self.assertRaises(RuntimeError):
            with preserve_runtime(model, (generator,)):
                torch.rand(3); torch.rand(3, generator=generator); random.random()
                model.eval(); raise RuntimeError("test")
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(torch.equal(loader, generator.get_state()))
        self.assertEqual(python, random.getstate())
        self.assertTrue(model.training); self.assertFalse(model[1].training)

    def test_hooks_do_not_change_forward_backward_or_rng(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            def forward(self, x): return self.model.layers[0](x)
        model = Tiny(); x = torch.randn(3, 4)
        y = model(x); y.sum().backward(); reference = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(); state = torch.get_rng_state()
        diag = LayerDiagnostics(model, True, 1, 4); diag.begin(1)
        actual = model(x); diag.close(); actual.sum().backward(); values = diag.gradients()
        self.assertTrue(torch.equal(y, actual)); self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for a, p in zip(reference, model.parameters()): self.assertTrue(torch.equal(a, p.grad))
        self.assertTrue(any("activation" in key for key in values)); self.assertTrue(any("gradient" in key for key in values))
        self.assertEqual(diag.handles, [])

    def test_domain_plan_deterministic_and_tiny_domains_retained(self):
        class Dataset:
            def __len__(self): return 21
            def reference(self, ref): return {"domain": "math" if ref < 1 else "en"}
        first = domain_plan(Dataset(), 4)
        self.assertEqual(first, domain_plan(Dataset(), 4)); self.assertEqual(first["math"], [0])
        self.assertEqual(len(set(first["en"])), 4)

    def test_combined_qwen_gdn_gate_projection_is_observed(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear_attn=torch.nn.Module()
                self.linear_attn.in_proj_ba=torch.nn.Linear(4,4,bias=False)
            def forward(self,x):return self.linear_attn.in_proj_ba(x)
        model=torch.nn.Module();model.model=torch.nn.Module()
        model.model.layers=torch.nn.ModuleList([Layer()])
        diag=LayerDiagnostics(model,True,1,4);diag.begin(1)
        model.model.layers[0](torch.ones(2,4));diag.close()
        self.assertIn('layers/model.layers.0.linear_attn.in_proj_ba/gdn_gate_preprojection_rms_sampled',diag.gradients())

    def test_outbox_pins_weights_after_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cp = root / "checkpoints/step_00002000"; cp.mkdir(parents=True)
            (cp / "model.safetensors").write_bytes(b"weight")
            (cp / "manifest.json").write_text(json.dumps({"step": 2000}))
            target = enqueue_checkpoint(root, cp, 2000, False, {})
            (cp / "model.safetensors").unlink()
            self.assertEqual((target / "checkpoint/model.safetensors").read_bytes(), b"weight")
            self.assertFalse((target / "checkpoint/optimizer.pt").exists())
            self.assertIn("abilities_pilot", json.loads((target / "request.json").read_text())["checks"])
            self.assertEqual(enqueue_checkpoint(root, cp, 2000, False, {}), target)

    def test_outbox_rejects_incorrect_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifest.json").write_text('{"step": 1}')
            with self.assertRaises(ValueError): enqueue_checkpoint(root, root, 2, False, {})

    def test_sharded_loader_iterator_preserves_checkpointed_generator(self):
        from small_agent.training.trainer import sharded_train_batches
        class Dataset(torch.utils.data.Dataset):
            def __len__(self): return 10
            def __getitem__(self, ref): return {"id": torch.tensor(ref)}
        generator = torch.Generator().manual_seed(41)
        state = generator.get_state().clone()
        batches = sharded_train_batches(Dataset(), range(10), 2, generator, {"num_workers": 0})
        self.assertEqual(next(batches)["id"].tolist(), [0, 1])
        self.assertTrue(torch.equal(state, generator.get_state()))

    def test_reference_panel_rejects_training_binding_drift(self):
        from small_agent.training.quality import ReferencePanel
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"manifest.json").write_text(json.dumps({"training_eligible":False,"tokenizer_sha256":"tokenizer","training_pack_manifest_sha256":"old"}))
            with self.assertRaisesRegex(ValueError,"binding"):
                ReferencePanel(root,"tokenizer","new")

    def test_optimized_formal_adoption_fails_closed(self):
        from small_agent.training.quality import validate_optimized_adoption
        validate_optimized_adoption({"distributed":{"communication_hook":"none"}})
        with self.assertRaisesRegex(ValueError,"receipt"):
            validate_optimized_adoption({"distributed":{"communication_hook":"bf16"}})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"receipt.json";path.write_text('{"explicit_adoption_authorized":false}')
            with self.assertRaisesRegex(ValueError,"incomplete"):
                validate_optimized_adoption({"distributed":{"communication_hook":"bf16"},"quality_adoption_receipt":str(path)})


if __name__ == "__main__": unittest.main()
