import sys
from pathlib import Path


def main(args):
    src_pth = Path(args[0])
    tgt_pth = Path(args[1])
    with open(src_pth / "wav.scp", "r") as f_full, open(
        src_pth / "utt2spk", "r"
    ) as f_utt2spk, open(src_pth / "utt2num_samples", "r") as f_numsamp, open(
        src_pth / "utt2uniq", "r"
    ) as f_uniq, open(
        args[2], "r"
    ) as f_clean:
        lines_full = f_full.readlines()
        lines_utt2spk = f_utt2spk.readlines()
        lines_numsamp = f_numsamp.readlines()
        lines_uniq = f_uniq.readlines()
        lines_clean = f_clean.readlines()

    # make utterance set
    utt_cln_set = set()
    for u_cln in lines_clean:
        u_cln = u_cln.strip()
        u_cln = "/".join([u_cln[:7], u_cln[8:-6], u_cln[-5:]])
        utt_cln_set.add(u_cln)
    print(f"# utts in clean set: {len(utt_cln_set)}")

    assert (
        len(lines_full) == len(lines_utt2spk) == len(lines_numsamp) == len(lines_uniq)
    )
    with open(tgt_pth / "wav.scp", "w") as f_out_wav, open(
        tgt_pth / "utt2spk", "w"
    ) as f_out_utt2spk, open(tgt_pth / "utt2num_samples", "w") as f_out_numsamp, open(
        tgt_pth / "utt2uniq", "w"
    ) as f_out_uniq:
        for u_full, u_utt2spk, u_numsamp, u_uniq in zip(
            lines_full, lines_utt2spk, lines_numsamp, lines_uniq
        ):
            uid = u_full.strip().split(" ")[0]
            if uid in utt_cln_set or uid[6:] in utt_cln_set:
                f_out_wav.write(u_full)
                f_out_utt2spk.write(u_utt2spk)
                f_out_numsamp.write(u_numsamp)
                f_out_uniq.write(u_uniq)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
