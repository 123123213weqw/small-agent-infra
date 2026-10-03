#!/usr/bin/env python3
"""Export a full training checkpoint as a local, inference-only HF model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.models import build_model, load_model_spec  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest["distributable"] or "unreviewed_candidate" not in run_manifest["purpose"]:
        raise ValueError("this exporter is restricted to the non-distributable candidate diagnostic")
    spec_path = Path(run_manifest["model_spec"]["path"])
    data_manifest_path = Path(run_manifest["data_manifest"]["path"])
    if sha256_file(spec_path) != run_manifest["model_spec"]["sha256"]:
        raise ValueError("model spec hash changed")
    if sha256_file(data_manifest_path) != run_manifest["data_manifest"]["sha256"]:
        raise ValueError("data manifest hash changed")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("license_gate", {}).get("quality_approved") is not False:
        raise ValueError("expected explicitly unreviewed candidate data")

    checkpoint = run / "checkpoints" / f"step_{args.checkpoint_step:08d}"
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if checkpoint_manifest["step"] != args.checkpoint_step:
        raise ValueError("checkpoint step mismatch")
    model_path = checkpoint / "model.safetensors"
    artifact = checkpoint_manifest["artifacts"]["model.safetensors"]
    if model_path.stat().st_size != artifact["bytes"] or sha256_file(model_path) != artifact["sha256"]:
        raise ValueError("checkpoint model hash mismatch")

    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    tokenizer_manifest = json.loads((args.tokenizer_dir / "manifest.json").read_text(encoding="utf-8"))
    if tokenizer_manifest["stage"] != "frozen_tokenizer":
        raise ValueError("tokenizer is not frozen")
    if sha256_file(tokenizer_path) != data_manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer hash differs from training data")
    if sha256_file(tokenizer_path) != tokenizer_manifest["artifacts"]["tokenizer.json"]["sha256"]:
        raise ValueError("tokenizer manifest hash mismatch")

    raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_ids = tokenizer_manifest["special_token_ids"]
    for token, expected_id in special_ids.items():
        if raw_tokenizer.token_to_id(token) != expected_id:
            raise ValueError(f"special token ID changed: {token}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=[
            token for token in special_ids if token not in {"<pad>", "<bos>", "<eos>", "<unk>"}
        ],
        model_max_length=8192,
        clean_up_tokenization_spaces=False,
    )
    if len(tokenizer) != data_manifest["vocab_size"]:
        raise ValueError("export tokenizer vocabulary size changed")
    probes = [
        "机器学习是一种方法。",
        "def add(a, b):\n    return a + b\n",
        "English text with  two spaces, Unicode π and $x^2$.\n",
    ]
    for probe in probes:
        expected = raw_tokenizer.encode(probe, add_special_tokens=False).ids
        if tokenizer.encode(probe, add_special_tokens=False) != expected:
            raise ValueError("HF tokenizer IDs differ from frozen tokenizer")
        if tokenizer.decode(expected, skip_special_tokens=False) != probe:
            raise ValueError("HF tokenizer roundtrip failed")

    spec = load_model_spec(spec_path)
    model = build_model(spec)
    missing, unexpected = load_model(model, str(model_path), strict=True)
    if missing or unexpected:
        raise ValueError(f"checkpoint model mismatch: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda:0")
    model = model.to(device, dtype=torch.bfloat16).eval()
    model.config.use_cache = True
    probe_ids = tokenizer.encode(probes[0], add_special_tokens=False)
    inputs = torch.tensor([probe_ids], device=device)
    with torch.inference_mode():
        original_logits = model(input_ids=inputs, use_cache=False).logits[:, -1].float().cpu()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(temporary)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        loaded_tokenizer = AutoTokenizer.from_pretrained(temporary, local_files_only=True, trust_remote_code=False)
        for probe in probes:
            if loaded_tokenizer.encode(probe, add_special_tokens=False) != raw_tokenizer.encode(
                probe, add_special_tokens=False
            ).ids:
                raise ValueError("reloaded tokenizer IDs differ")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            temporary, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
        ).to(device).eval()
        with torch.inference_mode():
            reloaded_logits = loaded_model(input_ids=inputs, use_cache=False).logits[:, -1].float().cpu()
        max_abs = float((original_logits - reloaded_logits).abs().max())
        if max_abs > 0.05:
            raise ValueError(f"HF export logits changed: max_abs={max_abs}")
        del loaded_model
        gc.collect()
        torch.cuda.empty_cache()

        (temporary / "README.md").write_text(
            "# Internal 1B diagnostic export\n\n"
            "This model was trained on quality-unreviewed candidate data for 50M tokens. "
            "It is not an approved or distributable pretrained Base model. "
            "Do not upload or treat benchmark scores as final model capability.\n",
            encoding="utf-8",
        )
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(temporary.iterdir())
            if p.is_file()
        }
        export_manifest = {
            "stage": "internal_non_distributable_hf_export",
            "checkpoint_step": args.checkpoint_step,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "data_manifest_sha256": sha256_file(data_manifest_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "dtype": "bfloat16",
            "reload_last_logits_max_abs": max_abs,
            "files": files,
        }
        (temporary / "export_manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        print(json.dumps({"output": str(output), "reload_last_logits_max_abs": max_abs}), flush=True)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
