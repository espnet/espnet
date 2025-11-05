# -*- coding: utf-8 -*-
"""
Log-Mel 同士を DTW でアラインし、可視化と簡易指標を出力するスクリプト
- 実音声(REF) と 合成音声(SYN) を読み込み
- Log-Mel を同一パラメータで抽出
- cdist(Scipy) + librosa.sequence.dtw でアライン
- 可視化: REF/SYN のLog-Melヒートマップ、コスト行列+DTWパス、フレーム差分推移、差分ヒートマップ
- 指標: フレーム毎L2差分の mean/median/95pct、(任意)MCD[dB]

使い方:
    python3 visualization_LogMel.py
（必要であれば main 下の設定を調整）
"""

import argparse
import math
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# ====== 入出力パス（固定で使う場合はここを書き換え） ======
REF_DEFAULT = "/Users/rikatarumi/Desktop/実験音声/sk045_1_0032.wav"  # 実音声
SYN_DEFAULT = "/Users/rikatarumi/Desktop/実験音声/utt1.wav"  # 合成音声
# SYN_DEFAULT = "/Users/rikatarumi/Desktop/実験音声/utt2.wav"

# ====== Log-Mel 抽出パラメータ（両者で完全一致させる） ======
SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 80
FMAX = 7600
POWER = 2.0  # パワースペクトログラム

# ====== 画像保存設定 ======
OUT_DIR = "./logmel_vis_out"
DPI = 180


def load_wav_mono(path: str, sr: int) -> np.ndarray:
    """WAVを読み込み、モノラル化＆所定サンプリングにリサンプル"""
    y, _sr = librosa.load(path, sr=sr, mono=True)  # librosaが自動でリサンプル/モノ化
    # 無音ガード（すべて0は避ける）
    if y.size == 0:
        raise ValueError(f"音声が空です: {path}")
    return y


def to_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """Log-Mel（dB）を返す。shape=(n_mels, T)"""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=POWER,
    )
    LM = librosa.power_to_db(S, ref=np.max)
    return LM  # (n_mels, T)


def dtw_align_indices(
    LM_ref: np.ndarray, LM_syn: np.ndarray, metric: str = "euclidean"
):
    """
    Log-Mel を DTW でアラインし、対応フレーム列とコスト行列を返す。
    - 入力: (n_mels, T)
    - 出力: idx_ref, idx_syn, C
        idx_ref, idx_syn: アライン後のフレームインデックス（同じ長さ）
        C: コスト行列 (T_ref, T_syn)
    """
    # cdistは (サンプル, 特徴量) を前提 → 転置
    A = LM_ref.T  # (T_ref, n_mels)
    B = LM_syn.T  # (T_syn, n_mels)

    # 特徴量次元一致チェック
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"メル次元が一致しません: ref={A.shape[1]} syn={B.shape[1]} "
            f"(n_mels={N_MELS}, fmin={FMIN}, fmax={FMAX}, n_fft={N_FFT}, hop={HOP_LENGTH} を揃えてください)"
        )

    # コスト行列（フレーム×フレーム）
    C = cdist(A, B, metric=metric)  # (T_ref, T_syn)

    # librosaのDTWは C を直接受け取れる
    D, wp = librosa.sequence.dtw(C=C)
    # 経路は終点→始点なので時系列順に反転
    wp = wp[::-1]  # shape=(L, 2), [:,0]がrefフレーム、[:,1]がsynフレーム

    idx_ref = wp[:, 0]
    idx_syn = wp[:, 1]
    return idx_ref, idx_syn, C


def align_and_diff(
    LM_ref: np.ndarray, LM_syn: np.ndarray, idx_ref: np.ndarray, idx_syn: np.ndarray
):
    """
    DTWの対応フレーム列に沿って、両者のLog-Melを整列し差分を計算
    - 返り値
        LM_ref_aligned: (n_mels, L)
        LM_syn_aligned: (n_mels, L)
        frame_l2: (L,)  フレームごとのL2距離（メル次元でL2）
        diff_map: (n_mels, L) メル周波数×時間の差分（SYN-REF）
    """
    # 対応フレームを取り出し（列選択）
    LM_ref_aligned = LM_ref[:, idx_ref]  # (n_mels, L)
    LM_syn_aligned = LM_syn[:, idx_syn]  # (n_mels, L)

    # フレームごとの差分（メル次元でのL2）
    # d_t = || ref[:,t] - syn[:,t] ||_2
    frame_l2 = np.linalg.norm(LM_ref_aligned - LM_syn_aligned, axis=0)

    # 差分ヒートマップ用（符号付き差分）
    diff_map = LM_syn_aligned - LM_ref_aligned
    return LM_ref_aligned, LM_syn_aligned, frame_l2, diff_map


def compute_mcd_db(
    y_ref: np.ndarray, y_syn: np.ndarray, sr: int, n_mfcc: int = 13
) -> float:
    """
    簡易MCD計算（DTW非考慮の全体平均）。必要に応じてアライン後フレームに対して使ってください。
    定義: MCD [dB] = (10 / ln10) * sqrt(2) * mean_t sqrt( sum_{d=1..K} (mc_ref[d,t] - mc_syn[d,t])^2 )
    ここでは 0次MFCC含む/含まないは用途で調整（librosaのmfccは0次含む）。実装は0..(n_mfcc-1)全部を使う。
    """
    # MFCC 抽出（同一設定を担保）
    mfcc_ref = librosa.feature.mfcc(
        y=y_ref, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_syn = librosa.feature.mfcc(
        y=y_syn, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # 長さを短い方に揃える（簡易）
    T = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
    X = mfcc_ref[:, :T]
    Y = mfcc_syn[:, :T]

    # フレームごとのユークリッド
    diff = X - Y
    euc = np.linalg.norm(diff, axis=0)  # (T,)

    c = (10.0 / np.log(10.0)) * np.sqrt(2.0)
    mcd = c * np.mean(euc)
    return float(mcd)


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def seconds_from_frames(frames: np.ndarray, hop: int, sr: int):
    return frames * hop / sr


def plot_all(
    ref_path: str,
    syn_path: str,
    LM_ref: np.ndarray,
    LM_syn: np.ndarray,
    idx_ref: np.ndarray,
    idx_syn: np.ndarray,
    C: np.ndarray,
    frame_l2: np.ndarray,
    diff_map: np.ndarray,
    out_dir: str,
):
    ensure_outdir(out_dir)

    # 1) REF Log-Mel
    fig1 = plt.figure(figsize=(9, 3))
    librosa.display.specshow(
        LM_ref,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"REF Log-Mel\n{Path(ref_path).name}")
    out1 = os.path.join(out_dir, "1_ref_logmel.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=DPI)
    plt.close(fig1)

    # 2) SYN Log-Mel
    fig2 = plt.figure(figsize=(9, 3))
    librosa.display.specshow(
        LM_syn,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"SYN Log-Mel\n{Path(syn_path).name}")
    out2 = os.path.join(out_dir, "2_syn_logmel.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=DPI)
    plt.close(fig2)

    # 3) コスト行列 + DTWパス
    fig3 = plt.figure(figsize=(5, 5))
    plt.imshow(C.T, origin="lower", aspect="auto")  # 軸の対応をわかりやすく
    plt.plot(idx_ref, idx_syn, linewidth=1.0)  # パス
    plt.xlabel("REF frame")
    plt.ylabel("SYN frame")
    plt.title("Cost matrix (cdist) with DTW path")
    out3 = os.path.join(out_dir, "3_cost_with_path.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=DPI)
    plt.close(fig3)

    # 4) フレームごとのL2差分の推移（横軸は秒）
    t_sec = seconds_from_frames(np.arange(len(frame_l2)), HOP_LENGTH, SR)
    fig4 = plt.figure(figsize=(9, 2.8))
    plt.plot(t_sec, frame_l2)
    plt.xlabel("Aligned time [s]")
    plt.ylabel("L2 diff (per frame)")
    plt.title("Frame-wise L2 difference (after DTW alignment)")
    out4 = os.path.join(out_dir, "4_framewise_l2.png")
    plt.tight_layout()
    plt.savefig(out4, dpi=DPI)
    plt.close(fig4)

    # 5) 差分ヒートマップ（SYN - REF）
    fig5 = plt.figure(figsize=(9, 3))
    librosa.display.specshow(
        diff_map,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
    )
    plt.colorbar()
    plt.title("Difference heatmap (SYN - REF) on aligned timeline")
    out5 = os.path.join(out_dir, "5_diff_heatmap.png")
    plt.tight_layout()
    plt.savefig(out5, dpi=DPI)
    plt.close(fig5)

    return out1, out2, out3, out4, out5


def summarize_stats(frame_l2: np.ndarray) -> dict:
    stats = {
        "mean": float(np.mean(frame_l2)),
        "median": float(np.median(frame_l2)),
        "p95": float(np.percentile(frame_l2, 95)),
        "max": float(np.max(frame_l2)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Log-Mel の DTW アライン＆可視化")
    parser.add_argument("--ref", type=str, default=REF_DEFAULT, help="実音声のwavパス")
    parser.add_argument(
        "--syn", type=str, default=SYN_DEFAULT, help="合成音声のwavパス"
    )
    parser.add_argument("--out", type=str, default=OUT_DIR, help="出力ディレクトリ")
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="cdistの距離（euclidean, cosine など）",
    )
    parser.add_argument(
        "--mcd", action="store_true", help="簡易MCD[dB]も計算して表示（参考値）"
    )
    args = parser.parse_args()

    ref_path = args.ref
    syn_path = args.syn
    out_dir = args.out

    print("== Settings ==")
    print(f"REF: {ref_path}")
    print(f"SYN: {syn_path}")
    print(
        f"SR={SR}, N_FFT={N_FFT}, HOP={HOP_LENGTH}, N_MELS={N_MELS}, FMIN={FMIN}, FMAX={FMAX}"
    )
    print(f"cdist metric={args.metric}")
    print()

    # 1) 音声読み込み
    y_ref = load_wav_mono(ref_path, SR)
    y_syn = load_wav_mono(syn_path, SR)

    # 2) Log-Mel 抽出
    LM_ref = to_logmel(y_ref, SR)  # (n_mels, T_ref)
    LM_syn = to_logmel(y_syn, SR)  # (n_mels, T_syn)
    print(f"LM_ref shape: {LM_ref.shape}  (n_mels, T_ref)")
    print(f"LM_syn shape: {LM_syn.shape}  (n_mels, T_syn)")

    # 3) DTW アライン
    idx_ref, idx_syn, C = dtw_align_indices(LM_ref, LM_syn, metric=args.metric)
    print(f"DTW path length: {len(idx_ref)} frames")

    # 4) 整列＆差分
    LM_ref_aln, LM_syn_aln, frame_l2, diff_map = align_and_diff(
        LM_ref, LM_syn, idx_ref, idx_syn
    )
    stats = summarize_stats(frame_l2)
    print("== Frame-wise L2 stats (after alignment) ==")
    print(
        f"mean={stats['mean']:.4f}, median={stats['median']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}"
    )

    # 5) 可視化保存
    outs = plot_all(
        ref_path,
        syn_path,
        LM_ref,
        LM_syn,
        idx_ref,
        idx_syn,
        C,
        frame_l2,
        diff_map,
        out_dir,
    )
    print("Saved figures:")
    for p in outs:
        print(" -", p)

    # 6) (任意) MCD[dB] — 参考用（DTW未考慮の簡易平均）
    if args.mcd:
        mcd = compute_mcd_db(y_ref, y_syn, SR, n_mfcc=13)
        print(f"[Ref] Simple (non-DTW) MCD ≈ {mcd:.3f} dB")

    # 7) サマリーをテキストでも保存（ポスター用の数値転記に便利）
    ensure_outdir(out_dir)
    summary_txt = os.path.join(out_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Frame-wise L2 stats (after DTW alignment)\n")
        f.write(f"mean={stats['mean']:.6f}\n")
        f.write(f"median={stats['median']:.6f}\n")
        f.write(f"p95={stats['p95']:.6f}\n")
        f.write(f"max={stats['max']:.6f}\n")
    print("Saved summary:", summary_txt)


if __name__ == "__main__":
    main()
