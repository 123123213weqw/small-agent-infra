"""Architecture factory kept separate from the training engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"factory", "expected_parameters", "config"}
    missing = required - spec.keys()
    if missing:
        raise ValueError(f"model spec is missing fields: {sorted(missing)}")
    return spec


def build_model(spec: dict[str, Any]):
    factory = spec["factory"]
    if factory == "qwen3_next":
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

        model = Qwen3NextForCausalLM(Qwen3NextConfig(**spec["config"]))
    else:
        raise ValueError(f"unknown model factory: {factory!r}")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(spec["expected_parameters"]):
        raise ValueError(
            f"parameter count changed: {parameters} != {spec['expected_parameters']}"
        )
    return model
