import sys
import time

import whisper

inp_file = sys.argv[1]
out_file = sys.argv[2]

model = whisper.load_model("base.en").to("cuda")

avg_times = []
with open(inp_file, "r") as f, open(out_file, "w") as g:
    for i, line in enumerate(f.readlines()):
        st_time = time.time()
        uid, path = line.strip().split(" ")
        try:
            result = model.transcribe(path)
            g.write(uid + " " + result["text"] + "\n")
        except:
            continue

        avg_times.append(time.time() - st_time)

        if i % 2000 == 0:
            print(f"i = {i} | Average Time: {sum(avg_times)/len(avg_times)}")
            avg_times = []
