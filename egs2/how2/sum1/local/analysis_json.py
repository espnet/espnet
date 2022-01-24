import json


with open("cascstats.json", "r") as f:
    cascstats = json.load(f)

with open("gtstats.json", "r") as f:
    gtstats = json.load(f)

with open("e2estats.json", "r") as f:
    e2estats = json.load(f)
with open("data/dev5_test_sum/text", "r") as f:
    vid2sum = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }

for key in e2estats.keys():
    diff_rougel = e2estats[key]["rougel"] - cascstats[key]["rougel"]
    diff_mtr = e2estats[key]["meteor"] - cascstats[key]["meteor"]
    if diff_rougel < -2 and diff_mtr < -2:
        print(
            "Example Key={} DiffM={} DiffR={}\n E2E: {}\nCASC:{} \n GT:{} \n REF:{} \n\n\n".format(
                key,
                diff_rougel,
                diff_mtr,
                e2estats[key]["hyp"],
                cascstats[key]["hyp"],
                gtstats[key]["hyp"],
                vid2sum[key],
            )
        )
