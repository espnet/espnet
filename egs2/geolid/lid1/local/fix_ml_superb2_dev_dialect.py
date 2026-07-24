indian_langs = ["tam", "tel", "guj"]

with open(f"data/dev_dialect_ml_superb2_lang/utt2lang", "r") as utt2lang_fp:
    utt2lang_lines = utt2lang_fp.readlines()

with open(f"data/dev_dialect_ml_superb2_lang/wav.scp", "r") as wav_scp_fp:
    wav_scp_lines = wav_scp_fp.readlines()

filtered_utt2lang_lines = []
utt2lang_origin = {}
for line in utt2lang_lines:
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
    filtered_utt2lang_lines.append(f"{uttid} {lang}\n")

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

with open(f"data/dev_dialect_ml_superb2_lang/utt2lang", "w") as utt2lang_fp:
    utt2lang_fp.writelines(sorted(filtered_utt2lang_lines))
with open(f"data/dev_dialect_ml_superb2_lang/wav.scp", "w") as wav_scp_fp:
    wav_scp_fp.writelines(sorted(filtered_wav_scp_lines))
