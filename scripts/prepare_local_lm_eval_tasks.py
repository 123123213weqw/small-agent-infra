#!/usr/bin/env python3
"""Mirror two official lm-eval task definitions to pinned local Parquet data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_task(source: Path, output: Path, task: str, hub_path: str, data_files: dict[str, Path]) -> None:
    original = source.read_text(encoding="utf-8")
    for old in (f"task: {task}", f"dataset_path: {hub_path}", "dataset_name: null" if task == "hellaswag" else "dataset_name: ARC-Easy"):
        if old not in original:
            raise ValueError(f"official task definition changed: missing {old!r}")
    renamed = f"local_{task}"
    original = original.replace(f"task: {task}", f"task: {renamed}", 1)
    original = original.replace(f"dataset_path: {hub_path}", "dataset_path: parquet", 1)
    if task == "arc_easy":
        original = original.replace("dataset_name: ARC-Easy", "dataset_name: null", 1)
    data_lines = ["dataset_kwargs:", "  data_files:"]
    for split, path in data_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        data_lines.append(f"    {split}: {json.dumps(str(path.resolve()))}")
    original = original.replace("output_type:", "\n".join(data_lines) + "\noutput_type:", 1)
    output.write_text(original, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing ai2_arc and hellaswag")
    parser.add_argument("--lm-eval-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["arc_easy", "hellaswag"], default=["arc_easy", "hellaswag"])
    args = parser.parse_args()
    data = args.data.resolve()
    package = args.lm_eval_package.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    arc = data / "ai2_arc" / "ARC-Easy"
    hella = data / "hellaswag" / "data"
    tasks = [
        (
            "arc_easy",
            "allenai/ai2_arc",
            package / "tasks/arc/arc_easy.yaml",
            {
                split: arc / f"{split}-00000-of-00001.parquet"
                for split in ("train", "validation", "test")
            },
        ),
        (
            "hellaswag",
            "Rowan/hellaswag",
            package / "tasks/hellaswag/hellaswag.yaml",
            {
                split: hella / f"{split}-00000-of-00001.parquet"
                for split in ("train", "validation")
            },
        ),
    ]
    files = {}
    for task, hub_path, source, data_files in tasks:
        if task not in args.tasks:
            continue
        dest = output / f"local_{task}.yaml"
        render_task(source, dest, task, hub_path, data_files)
        files[dest.name] = {"sha256": sha256_file(dest), "source_sha256": sha256_file(source)}
        for path in data_files.values():
            files[str(path.relative_to(data))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if "hellaswag" in args.tasks:
        source_utils = package / "tasks/hellaswag/utils.py"
        shutil.copy2(source_utils, output / "utils.py")
        files["utils.py"] = {"sha256": sha256_file(output / "utils.py")}
    manifest = {
        "stage": "local_mirror_of_official_lm_eval_tasks",
        "upstream_dataset_revisions": REVISIONS,
        "official_task_package": str(package),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "tasks": [f"local_{task}" for task in args.tasks]}))


if __name__ == "__main__":
    main()
