#!/usr/bin/env python3
"""Low-compute CUDA VRAM holder for explicitly reserved L40 cards."""

from __future__ import annotations

import argparse
import os
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gib", type=int, default=40)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match physical GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible to a holder process")
    if "L40" not in torch.cuda.get_device_name(0):
        raise RuntimeError("target is not an L40")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    requested = args.gib * 2**30
    if free_bytes < requested + 2**30:
        raise RuntimeError(
            f"GPU {args.physical_gpu}: insufficient free VRAM for a safe holder: "
            f"free={free_bytes}, requested={requested}"
        )
    held = torch.empty((requested,), dtype=torch.uint8, device="cuda:0")
    held[0] = 1
    held[-1] = 1
    torch.cuda.synchronize()
    print(f"holding physical GPU {args.physical_gpu}: {args.gib} GiB "
          f"({torch.cuda.get_device_name(0)}, total {total_bytes / 2**30:.1f} GiB)", flush=True)

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
