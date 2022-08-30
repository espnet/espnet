import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=argparse.FileType('r'))
    parser.add_argument('hypothesis', type=argparse.FileType('r'))
    args = parser.parse_args()

    ref = dict()
    for f in args.reference:
        words = f.strip().split(' ')
        ref[words[0]] = float(words[1])

    hyp = dict()
    for f in args.hypothesis:
        words = f.strip().split(' ')
        hyp[words[0]] = float(words[1])

    num = 0.0
    den = 0.0
    for utt, score in ref.items():
        den +=1.0
        if score < hyp[utt]:
            num+=1.0

    print("Search error is ", num*100/den)

if __name__ == '__main__':
    main()
