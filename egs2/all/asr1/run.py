import argparse
import glob
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

STAGE_DEFAULT = 11
STOP_STAGE_DEFAULT = 13
PREFIX = "asr_train_e_branchformer_size256_mlp1024_e12_mactrue_"
POSTFIX = "_raw_en_bpe500_sp"

INFERENCE_CONFIGS = {
    "auto_regressive": "conf/decode_asr.yaml",
    "ctc": "conf/inference/decode_mctc_itr0_thres0.9.yaml",
    "mask_ctc": "conf/inference/decode_mctc_itr10_thres0.99.yaml",
}


@dataclass(frozen=True)
class ModeConfig:
    config_dir: str
    inference_config: str
    inference_model: str
    is_auto_regressive: bool


MODE_PRESETS = {
    "ctc": ModeConfig(
        config_dir="ctc",
        inference_config=INFERENCE_CONFIGS["ctc"],
        inference_model="valid.cer_ctc.ave_10best.pth",
        is_auto_regressive=False,
    ),
    "auto_regressive": ModeConfig(
        config_dir="auto_regressive",
        inference_config=INFERENCE_CONFIGS["auto_regressive"],
        inference_model="valid.acc.ave_10best.pth",
        is_auto_regressive=True,
    ),
    "rnn_t": ModeConfig(
        config_dir="rnn_t",
        inference_config=INFERENCE_CONFIGS["auto_regressive"],
        inference_model="valid.acc.ave_10best.pth",
        is_auto_regressive=True,
    ),
    "mask_ctc": ModeConfig(
        config_dir="mask_ctc",
        inference_config=INFERENCE_CONFIGS["mask_ctc"],
        inference_model="valid.cer.ave_10best.pth",
        is_auto_regressive=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple ESPnet run.sh jobs from config presets."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_PRESETS.keys()),
        default=["mask_ctc"],
        help="Run presets to execute. Default keeps previous behavior: mask_ctc only.",
    )
    parser.add_argument("--stage", type=int, default=STAGE_DEFAULT)
    parser.add_argument("--stop-stage", type=int, default=STOP_STAGE_DEFAULT)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--pretrained-model", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--speed-perturb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for launched processes and return non-zero if any job fails.",
    )
    return parser.parse_args()


def infer_asr_stats_dir(speed_perturb: bool) -> str:
    if speed_perturb:
        return "exp/asr_stats_raw_en_bpe500_sp"
    return "exp/asr_stats_raw_en_bpe5000"


def infer_expdir(config_dir: str, pretrained_model: str | None) -> str:
    if pretrained_model is None:
        return f"exp/{config_dir}"
    pretrained_model_name = Path(pretrained_model).parent.name
    pretrained_model_name = pretrained_model_name.replace(PREFIX, "").replace(
        POSTFIX, ""
    )
    return f"exp/{config_dir}/init_by_{pretrained_model_name}"


def find_configs(config_dir: str) -> list[str]:
    configs = []
    for config_path in sorted(glob.glob(f"conf/{config_dir}/*.yaml")):
        if "compress" in config_path:
            continue
        configs.append(config_path)
    return configs


def build_base_command(
    preset: ModeConfig,
    stage: int,
    stop_stage: int,
    ngpu: int,
    resume: bool,
    speed_perturb: bool,
    pretrained_model: str | None,
) -> list[str]:
    cmd = [
        "./run.sh",
        "--stage",
        str(stage),
        "--ngpu",
        str(ngpu),
        "--stop_stage",
        str(stop_stage),
        "--inference_config",
        preset.inference_config,
        "--inference_asr_model",
        preset.inference_model,
        "--use_lm",
        "false",
        "--asr_stats_dir",
        infer_asr_stats_dir(speed_perturb),
        "--expdir",
        infer_expdir(preset.config_dir, pretrained_model),
    ]
    if not preset.is_auto_regressive:
        cmd += ["--use_maskctc", "true"]
    if not resume:
        cmd += ["--resume", "false"]
    if speed_perturb:
        cmd += ["--speed_perturb_factors", "0.9 1.0 1.1"]
    if pretrained_model is not None:
        cmd += ["--pretrained_model", pretrained_model]
    return cmd


def run_mode(
    preset: ModeConfig,
    stage: int,
    stop_stage: int,
    ngpu: int,
    resume: bool,
    speed_perturb: bool,
    pretrained_model: str | None,
    dry_run: bool,
) -> list[subprocess.Popen]:
    base_cmd = build_base_command(
        preset=preset,
        stage=stage,
        stop_stage=stop_stage,
        ngpu=ngpu,
        resume=resume,
        speed_perturb=speed_perturb,
        pretrained_model=pretrained_model,
    )
    configs = find_configs(preset.config_dir)
    if not configs:
        print(f"[WARN] No config found under conf/{preset.config_dir}/*.yaml")
        return []

    processes: list[subprocess.Popen] = []
    for config in configs:
        cmd = base_cmd + ["--asr_config", config]
        print("[RUN]", " ".join(shlex.quote(part) for part in cmd))
        if not dry_run:
            processes.append(subprocess.Popen(cmd))
    return processes


def main() -> int:
    args = parse_args()

    if args.pretrained_model is not None and not os.path.exists(args.pretrained_model):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")

    all_processes: list[subprocess.Popen] = []
    for mode in args.modes:
        preset = MODE_PRESETS[mode]
        processes = run_mode(
            preset=preset,
            stage=args.stage,
            stop_stage=args.stop_stage,
            ngpu=args.ngpu,
            resume=args.resume,
            speed_perturb=args.speed_perturb,
            pretrained_model=args.pretrained_model,
            dry_run=args.dry_run,
        )
        all_processes.extend(processes)

    print(f"Launched jobs: {len(all_processes)}")
    if args.dry_run or not args.wait:
        return 0

    return_code = 0
    for process in all_processes:
        process.wait()
        if process.returncode != 0:
            return_code = process.returncode
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

# Use pretrained_models

# #pretrained_model = "exp/baseline_ctc/asr_train_e_branchformer_size256_mlp1024_e12_mactrue_condition_ctc_raw_en_bpe500_sp/valid.cer_ctc.ave_10best.pth"
# pretrained_model = "exp/baseline_ctc/asr_train_e_branchformer_size256_mlp1024_e12_mactrue_inter_ctc_raw_en_bpe500_sp/valid.cer_ctc.ave_10best.pth"
# # condition mask ctc
# assert  os.path.exists(pretrained_model)
# condition_mask_ctc_jobs=0

# inference_config_list = [ i for i in glob.glob("conf/inference_proposed/*") ] if stage ==12 else [inference_configs["ctc"]]
# for pretrained_model in ["exp/baseline_ctc/asr_train_e_branchformer_size256_mlp1024_e12_mactrue_inter_ctc_raw_en_bpe500_sp/valid.cer_ctc.ave_10best.pth",
#                         "exp/baseline_ctc/asr_train_e_branchformer_size256_mlp1024_e12_mactrue_condition_ctc_raw_en_bpe500_sp/valid.cer_ctc.ave_10best.pth"]:
#         for inference_config in inference_config_list:
#             condition_mask_ctc_jobs += run(
#                 "proposed",
#                 inference_config,
#                 "valid.cer_ctc.ave_10best.pth",
#                 is_auto_regressive=False,
#                 pretrained_model=pretrained_model,
#                 )
# print(f"condition mask ctc jobs {condition_mask_ctc_jobs}")
