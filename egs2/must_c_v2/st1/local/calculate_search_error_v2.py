import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=argparse.FileType('r'))
    parser.add_argument('hypothesis', type=argparse.FileType('r'))
    parser.add_argument('ctc_weight', type=float)
    args = parser.parse_args()

    ref = dict()
    for f in args.reference:
        words = f.strip().split(' ')
        ref[words[0]] = (float(words[2]), float(words[3]))

    hyp = dict()
    for f in args.hypothesis:
        words = f.strip().split(' ')
        hyp[words[0]] = (float(words[2]), float(words[3]))

    num = 0.0
    den = 0.0
    for utt, (att, ctc) in ref.items():
        ref_score = ((1-args.ctc_weight) * att) + (args.ctc_weight * ctc)
        hyp_score = ((1-args.ctc_weight) * hyp[utt][0]) + (args.ctc_weight * hyp[utt][1])
        den +=1.0
        if ref_score < hyp_score:
            num+=1.0

    print("Search error is ", num*100/den)

if __name__ == '__main__':
    main()
