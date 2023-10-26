from typing import Dict

import numpy as np
from rich.progress import track

from espnet2.bin.asr_align import CTCSegmentation
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.dataset import kaldi_loader


def token2frames(
    speech: np.ndarray,
    text: str,
    aligner: CTCSegmentation,
    tokenizer: SentencepiecesTokenizer,
    converter: TokenIDConverter,
    token_frames: Dict[int, np.ndarray],
    sample_rate: int = 16000,
) -> Dict[int, np.ndarray]:
    bpe = tokenizer.text2tokens(text)

    ratio = aligner.estimate_samples_to_frames_ratio()
    task = aligner(speech, bpe)

    text_ids = converter.tokens2ids(task.text)

    for index, segment in enumerate(task.segments):
        duration = segment[1] - segment[0]
        frames = duration * sample_rate // ratio

        text_id = str(text_ids[index])

        if text_id not in token_frames:
            token_frames[text_id] = np.array([frames], dtype=np.int32)
        else:
            token_frames[text_id] = np.append(token_frames[text_id], frames)

    return token_frames


if __name__ == "__main__":
    aligner = CTCSegmentation(
        asr_train_config="./exp/asr_lr_0.001_warmp_25/config.yaml",
        asr_model_file="./exp/asr_lr_0.001_warmp_25/valid.acc.ave_10best.pth",
        kaldi_style_text=False,
        ngpu=1,
    )

    tokenizer = build_tokenizer(
        token_type="bpe",
        bpemodel="./data/en_token_list/tgt_bpe_unigram5000_ts_en/bpe.model",
    )
    converter = TokenIDConverter(
        token_list="./data/en_token_list/tgt_bpe_unigram5000_ts_en/tokens.txt"
    )
    token_frames = {}

    scp_loader = kaldi_loader("./dump/raw/train_clean_100_sp/wav.scp", "float32")
    with open("./dump/raw/train_clean_100_sp/text", "r", encoding="utf-8") as text_file:
        text_lines = text_file.readlines()

        for text_line in track(
            text_lines, description="Processing CTC-Segmentation..."
        ):
            text_line = text_line.strip().split()
            uid = text_line[0]

            text = " ".join(text_line[1:])
            speech = scp_loader[uid]

            token_frames = token2frames(
                speech=speech,
                text=text,
                aligner=aligner,
                converter=converter,
                tokenizer=tokenizer,
                token_frames=token_frames,
            )

    token_frames_path = "./exp/asr_stats_raw_rm_wavlm_large_21_km2000_bpe6000_bpe5000_sp/train/token_frames.npz"
    token_statistics_path = "./exp/asr_stats_raw_rm_wavlm_large_21_km2000_bpe6000_bpe5000_sp/train/token_statistics.npy"

    np.savez_compressed(token_frames_path, **token_frames)

    vocab_size = converter.get_num_vocabulary_size()
    token_statistics = np.ndarray((vocab_size, 3))  # mean, std, median
    for text_id in track(
        range(vocab_size), description="Processing token2statistics..."
    ):
        text = str(text_id)
        if text in token_frames:
            frames = token_frames[text]
            mean = np.mean(frames)
            median = np.median(frames)
            std = np.std(frames)
            token_statistics[text_id] = np.array([mean, std, median])
        else:
            token_statistics[text_id] = np.array([0, 1, 0])

    np.save(token_statistics_path, token_statistics)

    token_frames_data = np.load(token_frames_path)
    token_statistics_data = np.load(token_statistics_path)

    keys = list(token_frames_data.keys())
    for key in keys:
        print(f"{key=} {token_frames_data[key]=} {token_statistics[int(key)]=}")
