#!/usr/bin/env python3
"""Import generic scoring JSONL into an existing local Label Studio project.

Run this inside the Label Studio Python environment on the same machine as its
SQLite database. Predictions are never converted into human annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--user-id", type=int, required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def region(name: str, value: dict) -> dict:
    return {"from_name": name, "to_name": "sample", "type": "textarea" if name == "notes" else "choices", "value": value}


def to_prediction(annotation: dict) -> list[dict]:
    result = [
        region("decision", {"choices": [annotation["decision"]]}),
        region("quality", {"choices": [annotation["quality"]]}),
    ]
    if annotation.get("issues"):
        result.append(region("issues", {"choices": annotation["issues"]}))
    if annotation.get("reason"):
        result.append(region("notes", {"text": [annotation["reason"]]}))
    return result


def main() -> int:
    args = parse_args()
    os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = str(args.data_dir)
    os.environ["SENTRY_DISABLE"] = "1"
    from label_studio.server import _setup_env

    _setup_env()
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient
    from projects.models import Project
    from tasks.models import Prediction, Task

    user = get_user_model().objects.get(pk=args.user_id)
    project = Project.objects.get(pk=args.project_id)
    if user.active_organization_id != project.organization_id:
        raise ValueError("user and project are in different organizations")

    tasks = {}
    for task in Task.objects.filter(project=project).only("id", "data"):
        key = (task.data.get("source_file"), task.data.get("source_row_idx"))
        if key in tasks:
            raise ValueError(f"duplicate task source key {key}")
        tasks[key] = task

    results = {}
    prompt_hashes = set()
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") != "ok":
            continue
        key = (item["source"]["data.source_file"], item["source"]["data.source_row_idx"])
        if key in results:
            raise ValueError(f"duplicate scored result {key}")
        if key not in tasks:
            raise ValueError(f"result not found in project {key}")
        actual_hash = hashlib.sha256(tasks[key].data["text"].encode("utf-8")).hexdigest()
        if actual_hash != item["text_sha256"]:
            raise ValueError(f"text hash mismatch for {key}")
        prompt_hashes.add(item["prompt_sha256"])
        results[key] = item
    if len(prompt_hashes) != 1:
        raise ValueError(f"expected one prompt version; found {len(prompt_hashes)}")

    client = APIClient(HTTP_HOST="127.0.0.1")
    client.force_authenticate(user=user)
    pending = []
    for key, item in results.items():
        task = tasks[key]
        if Prediction.objects.filter(task=task, model_version=args.model_version).exists():
            continue
        pending.append((task, item))
    if args.limit:
        pending = pending[: args.limit]
    print(f"project_tasks={len(tasks)} scored={len(results)} pending={len(pending)}", flush=True)
    if args.dry_run:
        return 0

    created = 0
    for task, item in pending:
        response = client.post(
            "/api/predictions/",
            data={
                "task": task.id,
                "project": args.project_id,
                "model_version": args.model_version,
                "result": to_prediction(item["annotation"]),
            },
            format="json",
        )
        if response.status_code != 201:
            print(f"failed task={task.id} http={response.status_code} detail={str(response.data)[:300]}", file=sys.stderr)
            return 1
        created += 1
        if created % 25 == 0 or created == len(pending):
            print(f"created={created}/{len(pending)}", flush=True)
    print("verified_predictions", Prediction.objects.filter(project=project, model_version=args.model_version).count(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
