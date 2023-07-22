import os


for dname in ["tr_2000h_utt", "cv05_utt", "dev5_test_utt"]:
    with open(os.path.join("data", dname, "text"), "r") as f, open(
        os.path.join("data", dname, "segments"), "r"
    ) as g, open(os.path.join("data", dname, "transcript"), "w") as h:
        utt2text = {
            line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
            for line in f.readlines()
        }
        rec2transcript = {}
        for line in g.readlines():
            utt_id, dia_id, start_time, end_time = line.strip().split(" ")
            if dia_id not in rec2transcript:
                rec2transcript[dia_id] = [[utt_id, start_time, end_time]]
            else:
                rec2transcript[dia_id].append([utt_id, start_time, end_time])

        for rec, transcript in rec2transcript.items():
            transcript.sort(key=lambda x: float(x[1]))
            h.write(f"{rec} {'.'.join([utt2text[x[0]] for x in transcript])}\n")
