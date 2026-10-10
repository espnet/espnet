"""Build tiny CPU smoke-run manifests from real segment/diarize outputs.

Picks the shortest segments (within a duration window) spread over a few
recordings and writes them, in the same per-recording JSON manifest format
the `infer` stage consumes, to a sibling directory read by the
`conf/inference_smoke.yaml` / `conf/inference_oracle_smoke.yaml` configs.
See the "CPU smoke run" section of readme.md.

Usage:
  python src/make_smoke_manifests.py <mode> <src_root> <dst_root> \
      <n_recos> <n_per_reco> <min_sec> <max_sec>

where <mode> is `diarized` (src_root e.g. exp/eval/diarized) or `oracle`
(src_root e.g. exp/eval_oracle/segments; only segments with a
transcript are eligible). Only stdlib is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    mode, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    n_recos, n_per_reco = int(sys.argv[4]), int(sys.argv[5])
    lo, hi = float(sys.argv[6]), float(sys.argv[7])

    dev_src = src / "dev"
    manifests = sorted(dev_src.glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests under {dev_src}")

    # Collect candidate segments per recording.
    per_reco = {}
    for mf in manifests:
        data = json.loads(mf.read_text())
        cands = []
        for seg in data["segments"]:
            dur = float(seg["end"]) - float(seg["start"])
            if not (lo <= dur <= hi):
                continue
            if mode == "oracle" and not str(seg.get("text", "")).strip():
                continue
            cands.append((dur, seg))
        if cands:
            cands.sort(key=lambda t: t[0])
            per_reco[data["reco"]] = (mf, data, cands)

    if not per_reco:
        raise SystemExit(f"no segments in [{lo}, {hi}]s")

    # Pick the n_recos recordings with the most candidates (richest plumbing).
    chosen = sorted(per_reco, key=lambda r: -len(per_reco[r][2]))[:n_recos]

    out_dir = dst / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for reco in sorted(chosen):
        mf, data, cands = per_reco[reco]
        keep = [seg for _, seg in cands[:n_per_reco]]
        total += len(keep)
        out = {k: v for k, v in data.items() if k != "segments"}
        out["segments"] = keep
        (out_dir / mf.name).write_text(json.dumps(out, indent=1))
        durs = [float(s["end"]) - float(s["start"]) for s in keep]
        print(
            f"{reco}: kept {len(keep)} segs, {sum(durs):.1f}s audio "
            f"(durs {['%.2f' % d for d in durs]})"
        )
    print(f"wrote {len(chosen)} manifests, {total} segments -> {out_dir}")


if __name__ == "__main__":
    main()
