import sys
from collections import OrderedDict

import numpy as np
import torch
import yaml

from espnet2.torch_utils.device_funcs import to_device


def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(
            torch.from_numpy(v), p=2, dim=1
        ).squeeze()

    return embd_dic2


def load_yaml(yamlfile):
    with open(yamlfile, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


def main(args):
    org_scores = args[0]
    org_embds = args[1]
    cohort_embds = args[2]
    utt2spk = args[3]
    out_dir = args[4]
    cfg = load_yaml(args[5])
    ngpu = int(args[6])
    print("args,", args)
    print("cfg,", cfg)

    with open(org_scores) as f:
        org_scores = f.readlines()
    with open(utt2spk) as f:
        utt2spk = f.readlines()
    utt2spk = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in utt2spk
    }
    org_embds = load_embeddings(org_embds)
    cohort_embds = load_embeddings(cohort_embds)

    if cfg["average_spk"]:
        print(f"Averaging cohort embeddings per speaker")
        spk_embds_dic = {}
        for k, v in cohort_embds.items():
            spk = utt2spk[k]
            if spk not in spk_embds_dic:
                spk_embds_dic[spk] = []
            spk_embds_dic[spk].append(v)
        cohort_embds = spk_embds_dic

        new_cohort_embds = []
        for spk in cohort_embds:
            new_cohort_embds.append(torch.stack(cohort_embds[spk], dim=0).mean(0))
    else:
        new_cohort_embds = []
        for utt in cohort_embds:
            new_cohort_embds.append(cohort_embds[utt])
    cohort_embds = torch.stack(new_cohort_embds, dim=0)
    cohort_embds = to_device(cohort_embds, "cuda" if ngpu > 0 else "cpu")

    print(f"Cohort embeds size: {cohort_embds.size()}")
    if cohort_embds.size(0) < cfg["adaptive_cohort_size"]:
        print(
            f"Cohort size({cohort_embds.size(0)}) is smaller than"
            f" configured({cfg['adaptive_cohort_size']})."
            "Adjustint to cohort size"
        )
        cfg["adaptive_cohort_size"] = cohort_embds.size(0)

    with open(out_dir, "w") as f:
        for score in org_scores:
            utts, score, lab = score.strip().split(" ")
            enr, tst = utts.split("*")
            enr = to_device(org_embds[enr], "cuda" if ngpu > 0 else "cpu")
            tst = to_device(org_embds[tst], "cuda" if ngpu > 0 else "cpu")
            score = float(score)

            e_c = -1.0 * torch.cdist(enr, cohort_embds).mean(0)
            e_c = torch.topk(e_c, k=cfg["adaptive_cohort_size"])[0]

            e_c_m = torch.mean(e_c)
            e_c_s = torch.std(e_c)

            t_c = -1.0 * torch.cdist(tst, cohort_embds).mean(0)
            t_c = torch.topk(t_c, k=cfg["adaptive_cohort_size"])[0]

            t_c_m = torch.mean(t_c)
            t_c_s = torch.std(t_c)

            normscore_e = (score - e_c_m) / e_c_s
            normscore_t = (score - t_c_m) / t_c_s

            newscore = (normscore_e + normscore_t) / 2
            newscore = newscore.item()

            f.write(f"{utts} {newscore} {lab}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
