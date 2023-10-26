FIRST_CHINESE=ord(u"\u4E00")

if __name__ == "__main__":
    with open(
        "/ocean/projects/cis210027p/jiatong/discreate_asr/pseudo_label/wavlm_large/train_960/pseudo_labels_km1000.txt",
        "r", encoding="utf-8"
    ) as pseudo_label_file, open("data/local/860_text/text",
        "r", encoding="utf-8",
    ) as text_file, open("data/local/860_speech/speech",
        "w", encoding="utf-8",
    ) as speech_file:
    
        pseudo_labels = {}
        pseudo_label_lines = pseudo_label_file.readlines()
        for pseudo_label_line in pseudo_label_lines:
            pseudo_label_line = pseudo_label_line.strip()
            pseudo_label_line = pseudo_label_line.split()

            uid = pseudo_label_line[0]

            # id -> chinese character
            pseudo_ids = list(map(lambda label: chr(FIRST_CHINESE + int(label)), pseudo_label_line[1:]))
            pseudo_ids = " ".join(pseudo_ids)

            pseudo_labels[uid] = pseudo_ids

        text_lines = text_file.readlines()
        for text_line in text_lines:
            text_line = text_line.strip()
            text_line = text_line.split()

            uid = text_line[0]
            pseudo_label = pseudo_labels[uid]

            speech_file.write(f"{uid} {pseudo_label}\n")
