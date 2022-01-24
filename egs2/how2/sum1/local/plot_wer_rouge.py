import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open("cascade_output_stats", "r") as f:
    rouge_cascade = {
        line.strip().split(" ")[0]: float(line.strip().split(" ")[1])
        for line in f.readlines()
    }

with open("trim_output_stats", "r") as f:
    rouge_e2e = {
        line.strip().split(" ")[0]: float(line.strip().split(" ")[1])
        for line in f.readlines()
    }

with open("vid_wer_conformer", "r") as f:
    wer = {
        line.strip().split(" ")[0]: float(line.strip().split(" ")[1])
        for line in f.readlines()
        if float(line.strip().split(" ")[1]) <= 100
    }

with open("vid_shape", "r") as f:
    shape = {
        line.strip().split(" ")[0]: int(line.strip().split(" ")[1])
        for line in f.readlines()
    }


wers = np.array([v for _, v in wer.items()])
print(wers)
rougecas = np.array([rouge_cascade[k] for k, v in wer.items()])
rougee = np.array([rouge_e2e[k] for k, v in wer.items()])
shapes = np.array([shape[k] for k, v in wer.items()])
diff = rougee - rougecas
print(len(diff[diff < 0]), len(diff), len(diff[diff > 0]),len(diff[diff == 0]))
fig = plt.figure()
plt.scatter(x=wers, y=rougee - rougecas, label="CASCADE", marker="o")
plt.title("ROUGE-L Score versus WER")
plt.xlabel("WER")
plt.ylabel("ROUGE-L Score")
# plt.scatter(wers, rougee, label="E2E", marker="o")
plt.legend()
plt.savefig("rouge_v_wer.png")

# fig = plt.figure()
# plt.scatter(x=shapes, y=rougecas, label="CASCADE", marker="o")
# plt.scatter(x=shapes, y=rougee, label="E2E", marker="^")
# plt.xlabel("Input Frame Lengths")
# plt.ylabel("ROUGE-L Score")
# plt.title("ROUGE-L Score versus Lengths")
# plt.legend()
# plt.savefig("image2.png")
