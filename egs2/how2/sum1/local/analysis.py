import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nlgeval
from nlgeval import NLGEval
import editdistance

import json
from spacy.lang.en.stop_words import STOP_WORDS


freq_summary_words = [
    "in",
    "a",
    "this",
    "to",
    "free",
    "the",
    "video",
    "and",
    "learn",
    "from",
    "on",
    "with",
    "how",
    "tips",
    "for",
    "of",
    "expert",
    "an",
]
freq_trans_words = [
    "the",
    "to",
    "and",
    "you",
    "a",
    "it",
    "that",
    "of",
    "is",
    "i",
    "going",
    "we",
    "in",
    "your",
    "this",
    "’s",
    "so",
    "on",
]
word_remove = [
    "in",
    "this",
    "free",
    "learn",
    "how",
    "tips",
    "expert",
    "clip",
]
##    "video",

# STOP_WORDS.update(word_remove)
# word_to_remove = list(STOP_WORDS)


# def stopword_removal(line: str):
#    return line if line not in word_to_remove else ""


with open("data/dev5_test_concept/text", "r") as f:
    vid2concepts = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1:]
        for line in f.readlines()
    }

with open("data/dev5_test_sum/text", "r") as f:
    vid2sum = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }
    import spacy
    from spacy.lang.en import English

    nlp = spacy.load("en_core_web_sm")
    vid2sumphrases = {}
    vid2sumphrases = {
        key: " ".join([word.text for word in nlp(val).noun_chunks])
        for key, val in vid2sum.items()
    }


with open("text_devtest", "r") as f:
    vid2text = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }
    vid2phrases = {
        key: " ".join([word.text for word in nlp(val).noun_chunks])
        for key, val in vid2text.items()
    }

with open("dev5_test_pred_transcript.txt", "r") as f:
    vid2predtext = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        for line in f.readlines()
    }
    vid2predphrases = {
        key: " ".join([word.text for word in nlp(val).noun_chunks])
        for key, val in vid2predtext.items()
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
nlg = NLGEval()  # loads the models

e2estats = {}
with open("results/trim_e2e_pred.txt", "r") as g:
    for line2 in g.readlines():
        key = line2.strip().split(" ")[0]
        summ = " ".join(line2.strip().split(" ")[1:])
        ref = vid2sumphrases[
            key
        ]  # " ".join([word for word in nlp(vid2sum[key]).noun_chunks])
        # concepts = [word for word in line2.strip().split(" ")[1:] if word in ref]
        hyp = " ".join(
            [word.text for word in nlp(summ).noun_chunks]
        )  # " ".join(concepts) if len(concepts) > 1 else ""
        metrics_dict = (
            nlg.compute_individual_metrics([ref], hyp)
            if len(ref[0]) > 1 and len(hyp) > 1
            else {"METEOR": 0.0, "ROUGE_L": 0.0}
        )
        e2estats[key] = {
            "concept_mtr": metrics_dict["METEOR"],
            "concept_rougel": metrics_dict["ROUGE_L"],
            "concept_wer": float(editdistance.eval(ref, hyp)) / len(ref.split(" ")),
            "wer": wer[key] if key in wer else None,
            "shape": shape[key],
            "hyp": summ,
            "concept_hyp": hyp,
            "concept_ref": ref,
        }


with open("outputs/trim_e2e_pred.txt", "r") as f:
    for line1 in f.readlines():
        if "_1:" in line1:
            break
        else:
            key, rouge, meteor = line1.strip().split(" ")
            if key in e2estats:
                e2estats[key]["rougel"] = float(rouge) * 100
                e2estats[key]["meteor"] = float(meteor) * 100
            else:
                e2estats[key] = {
                    "rougel": float(rouge) * 100.0,
                    "meteor": float(meteor) * 100,
                }


cascstats = {}
with open("results/asrtext_bart_large_pred.txt", "r") as g:
    for line2 in g.readlines():
        key = line2.strip().split(" ")[0]
        summ = " ".join(line2.strip().split(" ")[1:])
        ref = vid2sumphrases[key]
        # ref = " ".join([word for word in nlp(vid2sum[key]).noun_chunks])
        ##vid2sumphrases[key]  # " ".join(vid2concepts[key])
        concepts = [word for word in line2.strip().split(" ")[1:] if word in ref]
        hyp = " ".join(
            [word.text for word in nlp(summ).noun_chunks]
        )  # " ".join(concepts) if len(concepts) > 1 else ""
        metrics_dict = (
            nlg.compute_individual_metrics([ref], hyp)
            if len(ref[0]) > 1 and len(hyp) > 1
            else {"METEOR": 0.0, "ROUGE_L": 0.0}
        )
        ref_trans = vid2phrases[key]
        hyp_trans = vid2predphrases[key]
        metrics_dict2 = (
            nlg.compute_individual_metrics([ref_trans], hyp_trans)
            if len(ref_trans[0]) > 1 and len(hyp_trans) > 1
            else {"METEOR": 0.0, "ROUGE_L": 0.0}
        )
        cascstats[key] = {
            "concept_mtr": metrics_dict["METEOR"] * 100,
            "concept_rougel": metrics_dict["ROUGE_L"] * 100,
            "concept_wer": float(editdistance.eval(ref, hyp)) / len(ref.split(" ")),
            "wer": wer[key] if key in wer else None,
            "shape": shape[key],
            "hyp": summ,
            "trans_mtr": metrics_dict2["METEOR"] * 100.0,
            "trans_rougel": metrics_dict2["ROUGE_L"] * 100.0,
            "concept_hyp": hyp,
            "concept_ref": ref,
            "trans_hyp": ref_trans,
            "hyp_trans": hyp_trans,
        }


with open("outputs/res_asrtext_bart_large_pred.txt", "r") as f:
    for line1 in f.readlines():
        if "_1:" in line1:
            break
        else:
            key, rouge, meteor = line1.strip().split(" ")
            if key in cascstats:
                cascstats[key]["rougel"] = float(rouge) * 100
                cascstats[key]["meteor"] = float(meteor) * 100
            else:
                cascstats[key] = {
                    "rougel": float(rouge) * 100.0,
                    "meteor": float(meteor) * 100,
                }

gtstats = {}
with open("results/groundtruthtext_bart_large_pred.txt", "r") as g:
    for line2 in g.readlines():
        key = line2.strip().split(" ")[0]
        summ = " ".join(line2.strip().split(" ")[1:])
        ref = vid2sumphrases[key]
        # ref = " ".join([word for word in nlp(vid2sum[key]).noun_chunks])
        # concepts = [word for word in line2.strip().split(" ")[1:] if word in ref]
        hyp = " ".join([word.text for word in nlp(summ).noun_chunks])
        metrics_dict = (
            nlg.compute_individual_metrics([ref], hyp)
            if len(ref[0]) > 1 and len(hyp) > 1
            else {"METEOR": 0.0, "ROUGE_L": 0.0}
        )
        gtstats[key] = {
            "concept_mtr": metrics_dict["METEOR"] * 100,
            "concept_rougel": metrics_dict["ROUGE_L"] * 100,
            "concept_wer": float(editdistance.eval(ref, hyp)) / len(ref.split(" ")),
            "wer": wer[key] if key in wer else None,
            "shape": shape[key],
            "hyp": " ".join(line2.strip().split(" ")[1:]),
            "concept_hyp": hyp,
            "concept_ref": ref,
        }


with open("outputs/res_groundtruthtext_bart_large_pred.txt", "r") as f:
    for line1 in f.readlines():
        if "_1:" in line1:
            break
        else:
            key, rouge, meteor = line1.strip().split(" ")
            if key in gtstats:
                gtstats[key]["rougel"] = float(rouge) * 100
                gtstats[key]["meteor"] = float(meteor) * 100
            else:
                gtstats[key] = {
                    "rougel": float(rouge) * 100.0,
                    "meteor": float(meteor) * 100,
                }


with open("cascstats2.json", "w") as f:
    json.dump(cascstats, f)

with open("gtstats2.json", "w") as f:
    json.dump(gtstats, f)

with open("e2estats2.json", "w") as f:
    json.dump(e2estats, f)


## GT v/s PRED TEXT
## 1. Check if noun phrases are missing in the transcript, then they are missing in the summary
keys = list(cascstats.keys())
trans_mtrs = [float(cascstats[key]["trans_mtr"]) for key in keys]
summ_mtrs = [float(cascstats[key]["meteor"]) for key in keys]
fig = plt.figure()
plt.scatter(x=summ_mtrs, y=trans_mtrs, label="CASCADE", marker="o")
plt.title("Summary Concept METEOR versus Transcript Concept METEOR")
plt.xlabel("Summary Concept METEOR")
plt.ylabel("Transcript Concept METEOR")
# plt.scatter(wers, rougee, label="E2E", marker="o")
plt.legend()
plt.savefig("trans_sum_meteor.png")

## E2E Model
# 1. Figure out if length constraint causes issues
keys = [key for key, val in e2estats.items() if val["shape"] >= 10000]
# print(vid2sumphrases[keys[0]], vid2concepts[keys[0]])
diff = np.array([e2estats[key]["rougel"] - cascstats[key]["rougel"] for key in keys])
print(len(diff[diff < 0]), len(diff), len(diff[diff > 0]), len(diff[diff == 0]))

# 1. Figure out if length constraint causes issues
keys = [key for key, val in e2estats.items() if val["shape"] < 10000]
# print(vid2sumphrases[keys[0]], vid2concepts[keys[0]])
diff = np.array([e2estats[key]["rougel"] - cascstats[key]["rougel"] for key in keys])
print(
    "ROUGE-L < 10k",
    len(diff[diff < 0]),
    len(diff),
    len(diff[diff > 0]),
    len(diff[diff == 0]),
)
## E2E v/s Cascade
# 1. Show how well noun phrases are retained in the summaries
print("ROUGE-L IMPROVEMENT ")
keys = list(e2estats.keys())
diff = [e2estats[key]["rougel"] - cascstats[key]["rougel"] for key in keys]
diff = np.array(
    [x for _, x in sorted(zip(keys, diff), key=lambda pair: pair[1], reverse=True)]
)
print(diff)
print(
    "ROUGE-L ALL ",
    len(diff[diff < 0]),
    len(diff),
    len(diff[diff > 0]),
    len(diff[diff == 0]),
)

keys = np.array(
    [x for x, _ in sorted(zip(keys, diff), key=lambda pair: pair[1], reverse=True)]
)
diff2 = np.array([e2estats[key]["meteor"] - cascstats[key]["meteor"] for key in keys])

for i, key in enumerate(keys[diff > 0][:10]):
    print(
        "{} RD={} MD={}\n {}\n {}\n{}\n{}\n{}\n{}\n\n".format(
            key,
            diff[i],
            diff2[i],
            e2estats[key],
            cascstats[key],
            e2estats[key]["hyp"],
            cascstats[key]["hyp"],
            gtstats[key]["hyp"],
            vid2sum[key],
        )
    )
for key in keys[diff < 0][:-10]:
    print(
        "{} RD={} MD={}\n {}\n {}\n{}\n{}\n{}\n{}\n\n".format(
            key,
            diff[i],
            diff2[i],
            e2estats[key],
            cascstats[key],
            e2estats[key]["hyp"],
            cascstats[key]["hyp"],
            gtstats[key]["hyp"],
            vid2sum[key],
        )
    )
# 2. Compare concept METEOR and METEOR Scores
print("METEOR IMPROVEMENT EXAMPLES\n\n\n")
keys = list(e2estats.keys())
diff = [e2estats[key]["meteor"] - cascstats[key]["meteor"] for key in keys]
diff = np.array(
    [x for _, x in sorted(zip(keys, diff), key=lambda pair: pair[1], reverse=True)]
)
keys = np.array(
    [x for x, _ in sorted(zip(keys, diff), key=lambda pair: pair[1], reverse=True)]
)

for key in keys[diff > 0][:10]:
    print(
        "{} {} {} \n{}\n{}\n{}\n\n".format(
            key,
            e2estats[key],
            cascstats[key],
            e2estats[key]["hyp"],
            cascstats[key]["hyp"],
            vid2sum[key],
        )
    )
for key in keys[diff < 0][:-10]:
    print(
        "{} {} {} \n{}\n{}\n{}\n\n".format(
            key,
            e2estats[key],
            cascstats[key],
            e2estats[key]["hyp"],
            cascstats[key]["hyp"],
            vid2sum[key],
        )
    )

# 3. Improvement in METEOR leads to IMP in ROUGE
keys = list(cascstats.keys())
diffm = np.array([e2estats[key]["meteor"] - cascstats[key]["meteor"] for key in keys])
diffr = np.array([e2estats[key]["rougel"] - cascstats[key]["rougel"] for key in keys])
fig = plt.figure()
plt.scatter(x=diffm, y=diffr)
plt.title("METEOR Difference versus ROUGE-L Difference")
plt.xlabel("Difference in METEOR")
plt.ylabel("Difference in ROUGE-L")
plt.legend()
plt.savefig("rouge_vs_meteor.png")