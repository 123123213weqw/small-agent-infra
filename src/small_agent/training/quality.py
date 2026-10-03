"""Opt-in, read-only training diagnostics and checkpoint evaluation outbox.

Statistics never replace the training objective, consume random numbers, or
modify tensors. All ranks take the same validation/collective control path.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from small_agent.evaluation import evaluate_causal_lm


class ReferencePanel(torch.utils.data.Dataset):
    """Document-level external panel; never merged into corpus validation PPL."""
    def __init__(self, directory, tokenizer_sha256, training_manifest_sha256):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["training_eligible"] or manifest["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("invalid external evaluation tokenizer/eligibility")
        if manifest["training_pack_manifest_sha256"] != training_manifest_sha256:
            raise ValueError("external panel contamination-screen binding differs")
        path = directory / "documents.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifacts"]["documents.jsonl"]["sha256"]:
            raise ValueError("external evaluation panel changed")
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.manifest = manifest

    def __len__(self): return len(self.rows)

    def reference(self, ref): return {"domain": self.rows[ref]["domain"], "id": self.rows[ref]["id"]}

    def __getitem__(self, ref):
        ids = torch.tensor(self.rows[ref]["input_ids"], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


@contextmanager
def preserve_runtime(model, generators=()):
    """Evaluation must not change the next training batch or RNG stream."""
    python = random.getstate()
    cpu = torch.get_rng_state()
    device = next(model.parameters()).device
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    states = [generator.get_state() for generator in generators]
    modes = [(module, module.training) for module in model.modules()]
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except ImportError:
        numpy_state = None
    try:
        yield
    finally:
        random.setstate(python)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
        for generator, state in zip(generators, states):
            generator.set_state(state)
        for module, mode in modes:
            module.training = mode
        if numpy_state is not None:
            np.random.set_state(numpy_state)


class HealthWindow:
    """Actual every-update statistics; reduce only at the logging boundary."""
    report_keys = ("loss_token_weighted", "predicted_tokens_window", "grad_norm_window_mean",
                   "grad_norm_window_max", "clip_fraction_window", "host_step_seconds_window_mean",
                   "host_step_seconds_window_max")
    def __init__(self, device, clip):
        if clip <= 0:
            raise ValueError("positive gradient clip required")
        self.device, self.clip = device, clip
        self.reset()

    def reset(self):
        # Loss sum, target count, norm sum, update count, clipped count, seconds.
        self.sums = torch.zeros(6, dtype=torch.float64, device=self.device)
        self.max_norm = torch.zeros((), dtype=torch.float64, device=self.device)
        self.max_seconds = 0.0

    def add(self, loss_sum, targets, grad_norm, seconds):
        if targets <= 0 or seconds < 0:
            raise ValueError("invalid health observation")
        norm = grad_norm.detach().double()
        self.sums[0] += loss_sum.detach().double()
        self.sums[1] += targets
        self.sums[2] += norm
        self.sums[3] += 1
        self.sums[4] += (norm > self.clip).double()
        self.sums[5] += seconds
        self.max_norm = torch.maximum(self.max_norm, norm)
        self.max_seconds = max(self.max_seconds, seconds)

    def report(self):
        sums, norm = self.sums.clone(), self.max_norm.clone()
        seconds = torch.tensor(self.max_seconds, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(norm, op=dist.ReduceOp.MAX)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        if sums[1] <= 0 or sums[3] <= 0:
            raise ValueError("empty health window")
        result = {
            "loss_token_weighted": float(sums[0] / sums[1]),
            "predicted_tokens_window": int(sums[1]),
            "grad_norm_window_mean": float(sums[2] / sums[3]),
            "grad_norm_window_max": float(norm),
            "clip_fraction_window": float(sums[4] / sums[3]),
            "host_step_seconds_window_mean": float(sums[5] / sums[3]),
            "host_step_seconds_window_max": float(seconds),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("non-finite health window")
        return result


class LayerDiagnostics:
    """Sampled rank-zero activations/gradients, not a recurrent-state monitor.

    Only attach hooks during a selected update, collect its first microbatch,
    and detach them before checkpoint recomputation/backward. GDN projection
    observations are explicitly raw gate inputs, not the actual retention gate.
    """
    def __init__(self, model, enabled=False, interval=200, maximum_elements=4096):
        if interval < 1 or maximum_elements < 1:
            raise ValueError("positive diagnostic interval and sample size required")
        self.model, self.enabled, self.interval = model, enabled, interval
        self.maximum_elements = maximum_elements
        self.handles, self.values = [], {}

    def begin(self, step):
        self.close()
        self.values = {}
        self.active = self.enabled and step % self.interval == 0
        if not self.active:
            return
        for name, module in self.model.named_modules():
            if name.startswith("model.layers.") and name.count(".") == 2:
                self.handles.append(module.register_forward_hook(self._hook(name, "activation")))
            elif name.endswith(("in_proj_a", "in_proj_b", "in_proj_ba")):
                self.handles.append(module.register_forward_hook(self._hook(name, "gdn_gate_preprojection")))

    def _hook(self, name, kind):
        def capture(module, inputs, output):
            key = f"layers/{name}/{kind}"
            if key in self.values:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            # Slice on device before casting; no full activation copy or RNG.
            flat = tensor.detach().reshape(-1)
            stride = max(1, (flat.numel() + self.maximum_elements - 1) // self.maximum_elements)
            sample = flat[::stride][:self.maximum_elements].float()
            self.values[key + "_rms_sampled"] = sample.square().mean().sqrt()
            self.values[key + "_absmax_sampled"] = sample.abs().max()
        return capture

    def gradients(self):
        if not self.active:
            return {}
        grouped = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or not name.startswith("model.layers."):
                continue
            layer = ".".join(name.split(".")[:3])
            flat = parameter.grad.detach().reshape(-1)
            stride = max(1, (flat.numel() + self.maximum_elements - 1) // self.maximum_elements)
            sample = flat[::stride][:self.maximum_elements].float()
            total, count = grouped.get(layer, (0, 0))
            grouped[layer] = total + sample.square().sum(), count + sample.numel()
        for layer, (total, count) in grouped.items():
            self.values[f"layers/{layer}/gradient_rms_sampled"] = (total / count).sqrt()
        result = {key: float(value) for key, value in self.values.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("non-finite sampled layer statistic")
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def domain_plan(dataset, maximum=128):
    groups = {}
    for ref in range(len(dataset)):
        domain = str(dataset.reference(ref)["domain"]) if hasattr(dataset, "reference") else "all"
        groups.setdefault(domain, []).append(ref)
    if maximum < 1:
        raise ValueError("positive quick validation size required")
    return {domain: refs if len(refs) <= maximum else [refs[i * len(refs) // maximum] for i in range(maximum)]
            for domain, refs in sorted(groups.items())}


def validate_optimized_adoption(run_config):
    """Formal BF16 communication/fusions require explicit, complete evidence."""
    optimized = run_config.get("distributed", {}).get("communication_hook", "none") != "none" or bool(run_config.get("kernel_optimizations"))
    if not optimized:
        return
    path = run_config.get("quality_adoption_receipt")
    if not path:
        raise ValueError("optimized formal training requires an explicit quality-adoption receipt")
    receipt = json.loads(Path(path).read_text())
    required = ("explicit_adoption_authorized", "paired_numeric_gates_passed", "eight_rank_resume_gate_passed", "abilities_and_generation_gate_passed")
    if not all(receipt.get(key) is True for key in required):
        raise ValueError("incomplete quality evidence; optimized formal training remains blocked")
    if receipt.get("model_spec_sha256") != hashlib.sha256(Path(run_config["model_spec"]).read_bytes()).hexdigest():
        raise ValueError("adoption evidence model binding differs")
    if receipt.get("approved_distributed") != run_config.get("distributed", {}) or receipt.get("approved_kernel_optimizations", {}) != run_config.get("kernel_optimizations", {}):
        raise ValueError("optimization configuration differs from approved evidence")


def quick_validation(model, dataset, plan, device, rank, world, batch_size=1, generators=()):
    started = time.monotonic()
    results = {}
    with preserve_runtime(model, generators):
        for domain, refs in plan.items():
            loader = DataLoader(Subset(dataset, refs[rank::world]), batch_size=batch_size,
                                shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            result = evaluate_causal_lm(model, loader, device)
            results[domain] = {"loss": result.loss, "ppl": result.perplexity,
                               "predicted_tokens": result.predicted_tokens, "samples": len(refs),
                               "small_sample_warning": len(refs) < 128}
    targets = sum(value["predicted_tokens"] for value in results.values())
    loss = sum(value["loss"] * value["predicted_tokens"] for value in results.values()) / targets
    return {"domains": results, "overall": {"loss": loss, "ppl": math.exp(min(loss, 80)),
            "predicted_tokens": targets}, "seconds": time.monotonic() - started}


def enqueue_checkpoint(run_dir, checkpoint, step, final, config):
    """Pin eval weights by hard link so normal checkpoint rotation is safe.

    An eval-only snapshot is not a resumable training checkpoint. It retains no
    optimizer. A dispatcher removes *only* these pins after successful scoring.
    """
    outbox = Path(run_dir) / "evaluation_requests"
    manifest = json.loads((Path(checkpoint) / "manifest.json").read_text())
    if manifest["step"] != step:
        raise ValueError("outbox step differs from committed checkpoint")
    outbox.mkdir(exist_ok=True)
    target = outbox / f"step_{step:08d}"
    if target.exists():
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".request-", dir=outbox))
    try:
        frozen = temporary / "checkpoint"
        frozen.mkdir()
        os.link(Path(checkpoint) / "model.safetensors", frozen / "model.safetensors")
        shutil.copy2(Path(checkpoint) / "manifest.json", frozen / "manifest.json")
        checks = ["generation"]
        if step % int(config.get("ability_interval_steps", 2000)) == 0 or final:
            checks.extend(["abilities_pilot", "code_pilot"])
        if final:
            checks.append("validation_full")
        request = {"version": "training_quality_outbox_v2", "step": step, "checks": checks,
                   "training_run": str(Path(run_dir).resolve()), "original_checkpoint": str(Path(checkpoint).resolve()),
                   "eval_only_not_resumable": True,
                   "checkpoint_manifest_sha256": hashlib.sha256((frozen / "manifest.json").read_bytes()).hexdigest(),
                   "config": config, "created_unix": time.time()}
        (temporary / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
