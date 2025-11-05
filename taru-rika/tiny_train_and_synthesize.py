import os
import re
import sys
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
print("This is REPO_ROOT!!!", REPO_ROOT);
EXP_ROOT  = REPO_ROOT / "egs2" / "jsut" / "tts1" / "exp"
RECIPE_DIR = REPO_ROOT / "egs2" / "jsut" / "tts1"
CONF_DIR   = RECIPE_DIR / "conf" / "tuning"
BASE_YAML  = CONF_DIR / "train_tacotron2.yaml"        # 既存のベース
TINY_YAML  = CONF_DIR / "train_tacotron2_tiny.yaml"   # 生成する極小学習用YAML
DEFAULT_TAG        = "tiny_demo"                               # exp ディレクトリ名の識別子
#TAG        = "tiny_demo"                               # exp ディレクトリ名の識別子
#TAG        = "tts_stats_raw_phn_tacotron_g2p_en_no_space"                               # exp ディレクトリ名の識別子
OUT_WAV    = REPO_ROOT / "taru-rika" / "tiny_demo.wav"
SAY_TEXT   = "落ちる"

def run(cmd, cwd=None, check=True):
    print(f"\n[RUN] {cmd}  (cwd={cwd or os.getcwd()})")
    subprocess.run(cmd, shell=True, cwd=cwd, check=check)

def ensure_tiny_yaml():
    """
    既存の train_tacotron2.yaml をベースに、
    ・max_epoch を 1（なければ追記）
    ・num_iters_per_epoch を 20（なければ追記）
    ・batch_size を 4（なければ追記）
    ・accum_grad を 1（なければ追記）
    など “最小限だけ学習” になるよう書き換えた YAML を生成
    """
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_YAML}")

    text = BASE_YAML.read_text()

    def upsert(key, val):
        nonlocal text
        pattern = rf"(?m)^\s*{re.escape(key)}\s*:\s*.*$"
        if re.search(pattern, text):
            text = re.sub(pattern, f"{key}: {val}", text)
        else:
            text += f"\n{key}: {val}\n"

    # 極小設定（デコーダ/スケジューラ等はそのまま、回数だけ絞る）
    upsert("max_epoch", 1)
    upsert("num_iters_per_epoch", 20)
    upsert("batch_size", 4)
    upsert("accum_grad", 1)
    # 早めに終わるよう学習サンプルのシャッフル/切り上げも軽めに
    upsert("num_workers", 2)

    TINY_YAML.write_text(text)
    print(f"[OK] wrote tiny config: {TINY_YAML}")

def stage_1_2_prepare():
    # LJSpeech のDLと前処理（自動）。CPU想定。
    run("./run.sh --stage 1 --stop-stage 2 --ngpu 0", cwd=RECIPE_DIR)

def stage_3_train_tiny():
    # 超短縮学習。--tag で expdir に識別子を付ける
    cmd = f"./run.sh --stage 3 --stop-stage 3 --ngpu 0 " \
          f"--train_config {TINY_YAML.relative_to(RECIPE_DIR)} --tag {DEFAULT_TAG}"
    run(cmd, cwd=RECIPE_DIR)

#def find_checkpoint():
#    # できあがった expdir を探索して、valid.loss.ave.pth か 最新の *.pth を拾う
#    exp_glob = RECIPE_DIR / "exp" / f"tts_train_*{TAG}*"
#    candidates = sorted(glob.glob(str(exp_glob)))
#    if not candidates:
#        # tag が exp 名に乗らないレシピ差異もあるので、fallback
#        candidates = sorted(glob.glob(str(RECIPE_DIR / "exp" / "tts_train_*")))
#    if not candidates:
#        raise FileNotFoundError("No expdir found under egs2/jsut/tts1/exp")
#
#    expdir = Path(candidates[-1])
#    print(f"[INFO] using expdir: {expdir}")
#
#    ckpt = expdir / "valid.loss.ave.pth"
#    if ckpt.exists():
#        return ckpt, expdir
#
#    # fallback: 最新の .pth
#    pths = sorted(expdir.glob("*.pth"))
#    if not pths:
#        # deeper search
#        pths = sorted(expdir.rglob("*.pth"))
#    if not pths:
#        raise FileNotFoundError("No checkpoint (.pth) found in expdir")
#
#    return pths[-1], expdir

def _latest_path(paths):
    """更新時刻(=最終更新)が新しい順で最後を返す。なければ None。"""
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def find_checkpoint(tag: Optional[str] = None) -> Tuple[Path, Path]:
    """
    チェックポイント(.pth / snapshot)と、その expdir を返す。
    - tts_train_*{TAG}* だけでなく、tts_*{TAG}*（例: tts_tiny_demo）も対象に含める
    - TAGが未指定なら DEFAULT_TAG を用いる
    - 代表的なファイル名の優先順位で探索
    """
    tag = tag or DEFAULT_TAG

    if not EXP_ROOT.exists():
        raise FileNotFoundError(f"EXP_ROOT not found: {EXP_ROOT}")

    # 1) 候補ディレクトリ: 典型の tts_train_* と TAGベース tts_*、さらに保険で *{TAG}*
    patterns = [
        f"tts_train_*{tag}*",
        f"tts_*{tag}*",
        f"*{tag}*",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(EXP_ROOT.glob(pat))
    # ディレクトリのみ、重複除去
    candidates = sorted({p for p in candidates if p.is_dir()})

    if not candidates:
        raise FileNotFoundError(
            "No expdir matched under "
            f"{EXP_ROOT}\n"
            f"  looked for patterns: {patterns}"
        )

    # 2) 候補ディレクトリごとに既知のチェックポイント名を探索
    ckpt_patterns = [
        "checkpoint.pth",
        "checkpoint.pth.tar",
        "valid.*.pth",
        "snapshot.ep.*",
        "*.pth",                # 保険（cls/自前保存など）
    ]

    # 最新更新ディレクトリ優先でスキャン
    candidates_sorted = sorted(
        candidates, key=lambda p: p.stat().st_mtime, reverse=True
    )

    for expdir in candidates_sorted:
        hits = []
        for pat in ckpt_patterns:
            hits.extend(sorted(expdir.rglob(pat)))
        if hits:
            ckpt = _latest_path(hits)
            return ckpt, expdir

    # 3) ここまで来たら「学習dirはあるが、保存物がまだ無い」可能性が高い
    with_logs = [d for d in candidates_sorted if list(d.rglob("train.log"))]
    if with_logs:
        raise FileNotFoundError(
            "Found training directories but no checkpoints yet.\n"
            f"  candidates: {[str(d) for d in candidates_sorted]}\n"
            "Likely causes:\n"
            "  - Training stage hasn’t run or produced a snapshot.\n"
            "  - Tiny config doesn’t save checkpoints (check epochs/save_interval).\n"
            "Actions:\n"
            "  - Run the training stage (e.g., --stage 7 --stop-stage 7).\n"
            "  - Ensure save_interval_epochs>=1 and keep_nbest_models>=1."
        )

    # 4) ログも無い＝実質まだ学習は未実行
    raise FileNotFoundError(
        "Candidate expdirs exist but contain no checkpoints or train.log.\n"
        f"  candidates: {[str(d) for d in candidates_sorted]}\n"
        "Please run training (stage) and verify outputs."
    )

def synthesize_with_python(ckpt_path):
    # exp 配下の最終 config.yaml を使う（conf/tuning ではない）
    exp_cfg = ckpt_path.with_name("config.yaml")

    # Stage 5 で生成された token_list（パスはあなたの環境の実パスに合わせてOK）
    tok = "/Users/rikatarumi/espnet/egs2/jsut/tts1/dump/token_list/phn_jaconv_pyopenjtalk/tokens.txt"

    # 一時テキストと出力ディレクトリ
    in_txt = REPO_ROOT / "_tmp_text.txt"
    out_dir = REPO_ROOT / "_tmp_tts_out"

    code = f"""
from pathlib import Path
import subprocess, sys, shutil

# 入力テキスト（先頭がキー、後ろが本文）
Path(r"{in_txt.as_posix()}").write_text("utt1 " + {SAY_TEXT!r} + "\\n", encoding="utf-8")

cmd = [
    sys.executable, "-m", "espnet2.bin.tts_inference",
    "--train_config", r"{exp_cfg.as_posix()}",
    "--model_file",   r"{ckpt_path.as_posix()}",
    "--data_path_and_name_and_type", r"{in_txt.as_posix()},text,text",
    "--output_dir",   r"{out_dir.as_posix()}",
    "--ngpu", "0",
]
print("[RUN]", " ".join(cmd))
subprocess.run(cmd, check=True)

# 生成された wav を探して既定の出力名にコピー
cands = list(Path(r"{out_dir.as_posix()}").rglob("*.wav"))
if not cands:
    print("ERROR: wav が見つかりません。出力ディレクトリ内を確認してください:", r"{out_dir.as_posix()}")
    sys.exit(3)

src_wav = sorted(cands)[0]
shutil.copy2(src_wav, r"{OUT_WAV.as_posix()}")
print("Saved:", r"{OUT_WAV.as_posix()}")
"""

    script = REPO_ROOT / "_tmp_synth.py"
    script.write_text(code)
    try:
        run(f"{sys.executable} {script.name}", cwd=REPO_ROOT)
    finally:
        try:
            script.unlink()
        except Exception:
            pass

#    # Python から合成（tacotron2 tiny の checkpoint + 既存ボコーダ）
#    # vocoder は Model Zoo の PWG(LJSpeech) を使用（自動DL）
#
#    tok = "/Users/rikatarumi/espnet/egs2/jsut/tts1/dump/token_list/phn_jaconv_pyopenjtalk/tokens.txt"
#
#    code = f"""
#import subprocess, sys
#cmd = [
#    sys.executable, "-m", "espnet2.bin.tts_inference",
#    "--train_config", r"{ckpt_path.with_name('config.yaml').as_posix()}",
#    "--model_file",   r"{ckpt_path.as_posix()}",
#    "--token_list",   r"{tok}",
#    "--token_type", "phn", "--cleaner", "jaconv", "--g2p", "pyopenjtalk",
#    "--text", r"{SAY_TEXT}",
#    "--out",  r"{OUT_WAV.as_posix()}",
#    "--device", "cpu",
#]
#print("[RUN]", " ".join(cmd))
#subprocess.run(cmd, check=True)
#print("Saved:", r"{OUT_WAV.as_posix()}")
#"""
##    code = f"""
##from pathlib import Path
##from espnet2.bin.tts_inference import Text2Speech
##import soundfile as sf
##
##token_list = Path("/Users/rikatarumi/espnet/egs2/jsut/tts1/dump/token_list/phn_jaconv_pyopenjtalk/tokens.txt")
##
##
##tts = Text2Speech(
##    train_config="{TINY_YAML.as_posix()}",
##    model_file="{ckpt_path.as_posix()}",
##    device="cpu",
##    token_list=str(token_list),
##    token_type="phn", cleaner="jaconv", g2p="pyopenjtalk",
##)
##out = tts("{SAY_TEXT}")
##wav, sr = out["wav"], out["fs"]
##sf.write("{OUT_WAV.as_posix()}", wav.numpy(), int(sr))
##print("Saved: {OUT_WAV.as_posix()}")
##"""
#    script = REPO_ROOT / "_tmp_synth.py"
#    script.write_text(code)
#    try:
#        run(f"{sys.executable} {script.name}", cwd=REPO_ROOT)
#    finally:
#        try:
#            script.unlink()
#        except Exception:
#            pass

def main():
    print("=== tiny train & synth demo (LJSpeech / Tacotron2 / CPU) ===")
    ensure_tiny_yaml()
    stage_1_2_prepare()
    stage_3_train_tiny()
    ckpt, expdir = find_checkpoint()
    print(f"[OK] checkpoint: {ckpt}")
    synthesize_with_python(ckpt)
    print(f"\n[DONE] listen: {OUT_WAV}")

if __name__ == "__main__":
    main()

