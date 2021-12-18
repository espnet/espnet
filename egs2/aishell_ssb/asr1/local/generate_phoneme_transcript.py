import argparse
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y',
            'z', 'zh']
INVALID_TONE = -1

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str)
args = parser.parse_args()


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    of = open(os.path.join(args.out_dir, 'phn_txt'), 'w', encoding='utf-8')

    unique_phones = set()
    # generate phoneme transcript
    with open(os.path.join(FILE_DIR, 'aishell-ssb-annotations.txt'), encoding='utf-8') as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0].split('.')[0]
            phones = tokens[2::2]

            cleaned_phones = []
            for p in phones:
                # 1. Separate initials and finals, and append initials to the list
                if p[:2] in INITIALS:
                    cleaned_phones.append(p[:2])
                    final = p[2:]
                elif p[:1] in INITIALS:
                    cleaned_phones.append(p[:1])
                    final = p[1:]
                else:
                    final = p

                # 2. Extract tones
                tone = final[-1]
                if tone.isnumeric():
                    tone = int(tone)
                    if tone == 5:  # neutral tone: 5 -> 0
                        tone = 0

                    final = final[:-1]
                else:
                    tone = INVALID_TONE

                # 2. Check for Erization
                erization = False
                if ('er' not in final and final[-1] == 'r') \
                        or 'uer' == final or 'ier' == final:
                    final = final.rstrip('r')
                    erization = True

                # 3. Format final: final + '_' + tone
                if tone != INVALID_TONE:
                    final = f'{final}_{tone}'

                # 4. Append final (and maybe erization)
                cleaned_phones.append(final)
                if erization:
                    cleaned_phones.append('er_0')

            # write
            phone_str = ' '.join(cleaned_phones)
            of.write(f'{utt}\t{phone_str}\n')

            # remember unique phones (without tone)
            for p in cleaned_phones:
                unique_phones.add(p.split('_')[0])

    of.close()

    # write a list of unique phones
    of = open(os.path.join(args.out_dir, 'phones.txt'), 'w', encoding='utf-8')
    unique_phones = sorted(list(unique_phones))
    for p in unique_phones:
        of.write(f'{p}\n')


if __name__ == "__main__":
    main()
