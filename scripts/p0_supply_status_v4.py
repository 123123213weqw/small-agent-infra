#!/usr/bin/env python3
"""Initialize/read rolling supply state; NOT a production worker or daemon."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.supply_control import SupplyControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    control = SupplyControl(args.database, recipe)
    try:
        out = control.status()
        out.update(target_train_body_tokens=recipe['target_train_body_tokens'],
            deployment_status=recipe['deployment_status'], production_daemon_started=False,
            existing_inventory_auto_imported=False,
            existing_inventory_explicitly_imported=out['legacy_imports'] > 0)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        control.close()


if __name__ == '__main__':main()
