indian_langs = ["tam", "tel", "guj"]

with open(f"data/dev_dialect_ml_superb2_lang/utt2spk", "r") as utt2spk_fp:
    utt2spk_lines = utt2spk_fp.readlines()

with open(f"data/dev_dialect_ml_superb2_lang/wav.scp", "r") as wav_scp_fp:
    wav_scp_lines = wav_scp_fp.readlines()

filtered_utt2spk_lines = []
utt2lang_origin = {}
for line in utt2spk_lines:
    uttid, lang = line.strip().split()
    utt2lang_origin[uttid] = lang
    if lang == "hin":
        for hin_lang in indian_langs:
            if hin_lang in uttid:
                lang = hin_lang
                uttid = uttid.replace("hin", lang)
                break
        else:
            raise ValueError(f"{indian_langs} not in {uttid}")
    filtered_utt2spk_lines.append(f"{uttid} {lang}\n")

filtered_wav_scp_lines = []
for line in wav_scp_lines:
    uttid, path = line.strip().split()
    if utt2lang_origin[uttid] == "hin":
        for hin_lang in indian_langs:
            if hin_lang in uttid:
                lang = hin_lang
                uttid = uttid.replace("hin", lang)
                break
        else:
            raise ValueError(f"{indian_langs} not in {uttid}")

    filtered_wav_scp_lines.append(f"{uttid} {path}\n")

with open(f"data/dev_dialect_ml_superb2_lang/utt2spk", "w") as utt2spk_fp:
    utt2spk_fp.writelines(sorted(filtered_utt2spk_lines))
with open(f"data/dev_dialect_ml_superb2_lang/wav.scp", "w") as wav_scp_fp:
    wav_scp_fp.writelines(sorted(filtered_wav_scp_lines))
