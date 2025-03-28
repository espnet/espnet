
from datasets import load_from_disk
from espnet2.bin.asr_inference import Speech2Text


if __name__ == "__main__":
    model = Speech2Text(
        "exp/debug/config.yaml",
        "exp/ls100_debug_ctc_utt/epoch83_step79968_valid.cer_ctc.ckpt"
    )

    ds = load_from_disk("/home/msomeki/workspace/librispeech_dataset")["dev-clean"]
    audio = ds[10]['audio']['array']
    transcript = model(audio)
    print(transcript[0][0])
    print(ds[10]['text'])
