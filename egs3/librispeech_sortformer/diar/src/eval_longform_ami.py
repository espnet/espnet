"""Long-form session-level diarization of AMI, writing per-session RTTMs.

Runs full-session (chunked + stitched) inference with the offline Sortformer on
each AMI session for a given mic condition (default SDM = Array1-01 = array1
mic1) and writes hypothesis + reference RTTMs. Score them with
``score_der_pyannote.py`` (run in the pyannote env) for collar-based DER.
"""

import argparse
import os

import lhotse  # noqa: E402
import numpy as np
import torch

from espnet2.diar.sortformer.convert_hf_sortformer import convert  # noqa: E402
from espnet2.diar.sortformer.convert_nest import load_nest_encoder  # noqa: E402
from espnet2.diar.sortformer.longform import (  # noqa: E402
    activity_to_segments,
    diarize_long,
    supervisions_to_rttm,
    write_rttm,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami", default="/raid/users/popcornell/AMI")
    ap.add_argument(
        "--cond", default="sdm", help="mic condition prefix, e.g. sdm/mdm/ihm-mix"
    )
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="/tmp/lf_ami_sdm_dev")
    ap.add_argument(
        "--ckpt", default=None, help="ESPnet .pth; default = convert HF pretrained"
    )
    ap.add_argument(
        "--nest", default=None, help="NEST encoder .pt to additionally load"
    )
    ap.add_argument(
        "--nemo_ckpt",
        default=None,
        help="NeMo full state_dict (.pt) of streaming v2; builds+loads v2 model",
    )
    ap.add_argument(
        "--mode",
        choices=["stitch", "streaming"],
        default="stitch",
        help="stitch=chunk+overlap-Hungarian; streaming=speaker-cache single pass",
    )
    ap.add_argument("--chunk_sec", type=float, default=90.0)
    ap.add_argument("--overlap_sec", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_sessions", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "hyp"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ref"), exist_ok=True)

    print("Building model ...", flush=True)
    if args.nemo_ckpt:
        from espnet2.diar.sortformer.convert_nemo_sortformer import convert_nemo

        model, rep = convert_nemo(args.nemo_ckpt, num_spk=4)
        print(
            "v2 convert: loaded=%d/%d missing=%s"
            % (rep["n_loaded"], rep["n_nemo"], rep["missing"][:4])
        )
    elif args.ckpt:
        from espnet2.diar.espnet_sortformer_model import build_sortformer_model

        model = build_sortformer_model(num_spk=4)
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
    else:
        model, _ = convert("nvidia/diar_sortformer_4spk-v1", num_spk=4)
    if args.nest:
        load_nest_encoder(model, args.nest)
    model.to(args.device).eval()

    rec = lhotse.load_manifest(
        os.path.join(args.ami, f"ami-{args.cond}_recordings_{args.split}.jsonl.gz")
    )
    sup = lhotse.load_manifest(
        os.path.join(args.ami, f"ami-{args.cond}_supervisions_{args.split}.jsonl.gz")
    )
    cuts = lhotse.CutSet.from_manifests(recordings=rec, supervisions=sup)
    ids = list(cuts.ids)
    if args.max_sessions:
        ids = ids[: args.max_sessions]

    for k, cid in enumerate(ids):
        cut = cuts[cid]
        wav = cut.load_audio()[0].astype(np.float32)
        if args.mode == "streaming":
            with torch.no_grad():
                p, plen = model.diarize_streaming(
                    torch.tensor(wav, device=args.device).unsqueeze(0)
                )
            preds = p[0, : plen[0]].float().cpu().numpy()
        else:
            preds = diarize_long(
                model,
                wav,
                sample_rate=cut.sampling_rate,
                chunk_sec=args.chunk_sec,
                overlap_sec=args.overlap_sec,
                device=args.device,
            )
        segs = activity_to_segments(preds, threshold=args.threshold)
        write_rttm(
            os.path.join(args.out, "hyp", f"{cut.recording_id}.rttm"),
            cut.recording_id,
            segs,
        )
        supervisions_to_rttm(
            os.path.join(args.out, "ref", f"{cut.recording_id}.rttm"),
            cut.recording_id,
            cut.supervisions,
        )
        n_ref_spk = len({s.speaker for s in cut.supervisions})
        print(
            f"[{k+1}/{len(ids)}] {cut.recording_id} dur={cut.duration/60:.1f}min "
            f"frames={preds.shape[0]} hyp_segs={len(segs)} ref_spk={n_ref_spk}",
            flush=True,
        )

    print(f"RTTMs written under {args.out}/{{hyp,ref}}", flush=True)


if __name__ == "__main__":
    main()
