#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def find_one(root: Path, patterns: list[str]) -> Path:
    matches = []
    for pat in patterns:
        matches.extend(root.rglob(pat))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find any of: {patterns} under {root}")
    if len(matches) > 1:
        # Prefer averaged/best checkpoint names if available.
        matches = sorted(
            matches,
            key=lambda p: (
                0 if "valid" in p.name and p.suffix == ".pth" else 1,
                str(p),
            ),
        )
    return matches[0]


def sanitize_train_config(
    cfg: dict,
    feats_stats: str,
    max_epoch: int,
    num_iters_per_epoch: int,
    batch_size: int,
    valid_batch_size: int,
    accum_grad: int,
    use_amp: bool,
) -> dict:
    # Remove HF/model-card metadata and recipe-runtime fields.
    remove_keys = {
        "config",
        "required",
        "version",
        "print_config",
        "dry_run",
        "distributed",
        "dist_init_method",
        "local_rank",
        "dist_master_addr",
        "dist_master_port",
        "output_dir",
        "train_data_path_and_name_and_type",
        "valid_data_path_and_name_and_type",
        "train_shape_file",
        "valid_shape_file",
        "collect_stats",
        "write_collected_feats",
        "token_list",
        "token_type",
        "bpemodel",
        "use_preprocessor",
        "non_linguistic_symbols",
        "cleaner",
        "g2p",
        "init_param",
        "ignore_init_mismatch",
        "pretrain_path",
    }
    for k in remove_keys:
        cfg.pop(k, None)

    # Local fine-tune defaults. Users can override them from run.sh.
    cfg["max_epoch"] = max_epoch
    if num_iters_per_epoch > 0:
        cfg["num_iters_per_epoch"] = num_iters_per_epoch
    else:
        cfg.pop("num_iters_per_epoch", None)
    cfg["batch_size"] = batch_size
    cfg["valid_batch_size"] = valid_batch_size
    cfg["accum_grad"] = accum_grad
    cfg["drop_last_iter"] = False
    cfg["use_amp"] = use_amp
    cfg["num_workers"] = 1
    cfg["log_interval"] = 1
    cfg["num_att_plot"] = 0
    cfg["keep_nbest_models"] = [1]
    cfg["nbest_averaging_interval"] = 0
    cfg["patience"] = None

    # Avoid cluster/distributed-only settings.
    cfg["dist_launcher"] = None
    cfg["multiprocessing_distributed"] = False
    cfg["dist_world_size"] = 1
    cfg["dist_rank"] = 0
    cfg["sharded_ddp"] = False
    cfg["ddp_comm_hook"] = "none"
    cfg["unused_parameters"] = False
    cfg["gradient_as_bucket_view"] = True

    # CPU/local-friendly optimizer settings.
    if isinstance(cfg.get("optim_conf"), dict):
        cfg["optim_conf"]["lr"] = 1.0e-5
        cfg["optim_conf"]["fused"] = False

    cfg["scheduler"] = "warmuplr"
    cfg["scheduler_conf"] = {"warmup_steps": 1}

    # Avoid optional Flash Attention dependency.
    for key in ["encoder_conf", "decoder_conf"]:
        if isinstance(cfg.get(key), dict):
            cfg[key]["use_flash_attn"] = False
            cfg[key]["gradient_checkpoint_layers"] = []

    cfg["normalize"] = "global_mvn"
    cfg["normalize_conf"] = {"stats_file": feats_stats}

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="espnet/owsm_v4_small_370M")
    parser.add_argument("--out_dir", default="downloads/owsm_v4_small_370M")
    parser.add_argument("--token_dir", default="data/de_token_list/bpe_unigram50000")
    parser.add_argument("--train_config", default="conf/finetune_owsm_v4_small.yaml")
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--num_iters_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--accum_grad", type=int, default=8)
    parser.add_argument(
        "--use_amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    snapshot = Path(snapshot_download(repo_id=args.repo_id))
    out_dir = Path(args.out_dir)
    token_dir = Path(args.token_dir)
    train_config = Path(args.train_config)

    out_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    train_config.parent.mkdir(parents=True, exist_ok=True)

    model_pth = find_one(snapshot, ["*.pth"])
    config_yaml = find_one(snapshot, ["config.yaml"])
    bpe_model = find_one(snapshot, ["bpe.model"])
    feats_stats = find_one(snapshot, ["feats_stats.npz"])

    # Use symlinks to avoid copying huge files into the recipe workspace.
    links = {
        out_dir / "model.pth": model_pth,
        out_dir / "config.yaml": config_yaml,
        out_dir / "bpe.model": bpe_model,
        out_dir / "feats_stats.npz": feats_stats,
        token_dir / "bpe.model": bpe_model,
    }
    for dst, src in links.items():
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    cfg = yaml.safe_load(config_yaml.read_text())
    token_list = cfg["token_list"]
    (token_dir / "tokens.txt").write_text(
        "\n".join(token_list) + "\n", encoding="utf-8"
    )

    sanitized = sanitize_train_config(
        cfg,
        feats_stats=str((out_dir / "feats_stats.npz").as_posix()),
        max_epoch=args.max_epoch,
        num_iters_per_epoch=args.num_iters_per_epoch,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        accum_grad=args.accum_grad,
        use_amp=args.use_amp,
    )
    train_config.write_text(
        yaml.safe_dump(sanitized, sort_keys=False, allow_unicode=True)
    )

    print("Prepared OWSM assets")
    print("repo_id:", args.repo_id)
    print("snapshot:", snapshot)
    print("model:", out_dir / "model.pth")
    print("bpe_model:", token_dir / "bpe.model")
    print("tokens:", token_dir / "tokens.txt", "num_tokens=", len(token_list))
    print("train_config:", train_config)


if __name__ == "__main__":
    main()
