#!/usr/bin/env python3
"""Read one API credential locally on its host, without shell-sourcing the file.

The worker still verifies the expected account fingerprint and shared budget.
This wrapper neither prints the credential nor places it in command arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
import os
from pathlib import Path
import sys


def load_key(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        if line.startswith("DEEPSEEK_API_KEY="):
            values.append(line.split("=", 1)[1].strip().strip("\"'"))
    if len(values) != 1 or not values[0] or any(c.isspace() for c in values[0]):
        raise ValueError("expected one nonempty credential, without whitespace")
    return values[0]


def load_profile(path: Path):
    """A private host profile references a credential; never contains its value."""
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("version") != 1 or profile.get("api_base") != "https://api.deepseek.com":
        raise ValueError("unsupported profile or non-official API endpoint")
    if profile.get("model") != "deepseek-flash":
        raise ValueError("unexpected reviewer model")
    root = Path(profile["worker_code_root"]).resolve()
    required = {"scripts/p0_curation_worker_v3.py", "scripts/p0_curation_v3.py",
                "src/small_agent/curation/core.py", "src/small_agent/curation/store.py",
                "src/small_agent/curation/ledger.py"}
    files = profile["worker_files_sha256"]
    if not required <= set(files):
        raise ValueError("incomplete existing-worker identity")
    for name, expected in files.items():
        candidate = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or root not in candidate.resolve().parents:
            raise ValueError("unsafe worker identity path")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError("existing worker code changed")
    credential = Path(profile["credentials_env"])
    if credential.stat().st_mode & 0o077:
        raise ValueError("credential file must not be group/world accessible")
    key = load_key(credential)
    if hashlib.sha256(key.encode()).hexdigest() != profile["expected_key_sha256"]:
        raise ValueError("configured account changed; credential never printed")
    return profile, key, root / "scripts/p0_curation_worker_v3.py"


def argument(rest, flag):
    if rest.count(flag) != 1:
        raise ValueError("one explicit " + flag + " required")
    i = rest.index(flag)
    if i + 1 >= len(rest):
        raise ValueError("missing " + flag + " value")
    return rest[i + 1]


def profile_worker_args(profile: dict, rest: list[str]) -> list[str]:
    """No paid execution merely because an account has a large balance."""
    cap = Decimal(str(profile["campaign_budget_cny"]))
    budget = Decimal(argument(rest, "--budget-cny"))
    if not cap.is_finite() or not budget.is_finite() or cap <= 0 or not 0 < budget <= cap:
        raise ValueError("campaign budget not authorized or exceeded")
    if "--allow-api" not in rest:
        raise ValueError("explicit --allow-api required")
    if Path(argument(rest, "--ledger")).resolve() != Path(profile["campaign_ledger"]).resolve():
        raise ValueError("use the profile shared campaign ledger")
    if argument(rest, "--judge") != "deepseek":
        raise ValueError("profile is only for the DeepSeek reviewer")
    batch = Path(argument(rest, "--batch"))
    judge = json.loads((batch / "judges.json").read_text())["judges"]["deepseek"]
    if judge["model"] != profile["model"] or judge["endpoint"]["base_url"].rstrip("/") != profile["api_base"]:
        raise ValueError("batch provider differs from configured profile")
    if judge["endpoint"]["kind"] != "responses" or judge["generation"]["thinking"] != "disabled" or judge["generation"]["temperature"] != 0:
        raise ValueError("unexpected reviewer protocol/generation settings")
    if judge["generation"]["max_tokens"] > profile["max_output_tokens"]:
        raise ValueError("batch output cap exceeds profile")
    if "--expected-key-sha256" in rest:
        if argument(rest, "--expected-key-sha256") != profile["expected_key_sha256"]:
            raise ValueError("supplied account binding differs")
    else:
        rest = [*rest, "--expected-key-sha256", profile["expected_key_sha256"]]
    return rest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--credentials-env", type=Path)
    group.add_argument("--profile", type=Path, help="private host configuration referencing the existing worker")
    p.add_argument("--check-profile", action="store_true", help="offline identity check only; zero model requests")
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = a.worker_args
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if a.profile:
        profile, key, worker = load_profile(a.profile)
        if a.check_profile:
            if rest:
                p.error("profile check does not accept worker arguments")
            print(json.dumps({"stage": "profile_identity_verified_not_live_generation_test",
                "model": profile["model"], "api_base": profile["api_base"],
                "existing_worker": str(worker), "campaign_budget_cny": profile["campaign_budget_cny"],
                "key_binding_verified": True, "new_model_requests": 0, "training_started": False}))
            return
        rest = profile_worker_args(profile, rest)
    else:
        if a.check_profile:
            p.error("--check-profile requires --profile")
        key = load_key(a.credentials_env)
        worker = Path(__file__).with_name("p0_curation_worker_v3.py")
    if not rest:
        p.error("worker arguments required after --")
    os.environ["DEEPSEEK_API_KEY"] = key
    os.execv(sys.executable, [sys.executable, str(worker), *rest])


if __name__ == "__main__":
    main()
