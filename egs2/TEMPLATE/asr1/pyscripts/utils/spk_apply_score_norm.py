import sys
import yaml
import torch
import numpy as np
from collections import OrderedDict

def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    #keys = list(embd_dic.keys())
    #values = torch.from_numpy(np.stack(list(embd_dic.values()), axis=0))
    #values = torch.nn.functional.normalize(values, p=2, dim=1).numpy()
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(torch.from_numpy(v), p=2, dim=1).squeeze().numpy()


    #return {k: v for k, v in zip(keys, values)}
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
    print(args)
    print(cfg)

    with open(org_scores) as f:
        org_scores = f.readlines()
    with open(utt2spk) as f:
        utt2spk = f.readlines()
    utt2spk = {line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in utt2spk}
    org_embds = load_embeddings(org_embds)
    cohort_embds = load_embeddings(cohort_embds)

    if cfg["average_spk"]:
        spk_embds_dic = {}
        for k, v in cohort_embds.items():
            spk = utt2spk[k]
            if spk not in spk_embds_dic:
                spk_embds_dic[spk] = []
            spk_embds_dic[spk].append(v)
        for spk in spk_embds_dic:
            spk_embds_dic[spk] = np.mean(spk_embds_dic[spk], axis=0)
        cohort_embds = spk_embds_dic
    cohort_embds = torch.from_numpy(np.array(list(cohort_embds.values())))
    print(f"{cohort_embds.size()}")

    with open(out_dir, "w") as f:
        for score in org_scores:
            utts, score, lab = score.strip().split(" ")
            enr, tst = utts.split("*")
            enr = torch.from_numpy(org_embds[enr])
            tst = torch.from_numpy(org_embds[tst])
            score = float(score)

            e_c = -1. * torch.cdist(enr, cohort_embds).mean(0)
            e_c = torch.topk(e_c, k=cfg["adaptive_cohort_size"])[0]

            e_c_m = torch.mean(e_c)
            e_c_s = torch.std(e_c)

            t_c = -1. * torch.cdist(tst, cohort_embds).mean(0)
            t_c = torch.topk(t_c, k=cfg["adaptive_cohort_size"])[0]

            t_c_m = torch.mean(t_c)
            t_c_s = torch.std(t_c)

            normscore_e = (score - e_c_m) / e_c_s
            normscore_t = (score - t_c_m) / t_c_s

            newscore = (normscore_e + normscore_t) / 2
            newscore = newscore.item()
            #print("score", score, "newscore", newscore, "label", lab)

            f.write(f"{utts} {newscore} {lab}\n")
            stat_dic = {
                "e_c_m": e_c_m,
                "e_c_s": e_c_s,
                "t_c_m": t_c_m,
                "t_c_s": t_c_s
            }

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
