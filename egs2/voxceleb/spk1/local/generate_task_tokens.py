import sys

assert len(sys.argv) == 4
utt2category_files = sys.argv[1]
task_names = sys.argv[2]
output_file = sys.argv[3]

task_names = task_names.split(",")
task_dict = dict()
cnt = 0
for name in sorted(task_names):
    task_dict[name] = cnt
    cnt += 1

utt2category_files = utt2category_files.split(",")

assert len(utt2category_files) == len(task_names)
with open(output_file, "w") as f:
    for tn, utt2cat in zip(task_names, utt2category_files):
        task_token = task_dict[tn]

        with open(utt2cat, "r") as f2:
            utts = f2.readlines()
        for utt in utts:
            utt_id = utt.strip().split(" ")[0]
            f.write(f"{utt_id} {task_token}\n")
