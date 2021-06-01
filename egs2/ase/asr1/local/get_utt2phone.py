import sys


def main():
    if len(sys.argv) < 3:
        print(
            "get_utt2phone.py: Requires two arguments:\n"
            "\tpython get_utt2phone.py <text-phone> <utt2phone>"
        )
        exit(1)
    text_phone = sys.argv[1]
    utt2phone = sys.argv[2]

    with open(utt2phone, 'w') as f:
        f.write(get_utt2phone(text_phone))


def get_utt2phone(text_phone: str) -> str:
    ret = ''

    u2p_dict = {}
    with open(text_phone) as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            assert len(tokens) > 1
            utt = tokens[0]
            trans = " ".join(tokens[1:])

            assert '.' in utt
            utt, time = utt.split('.')
            time = int(time)
            if utt not in u2p_dict:
                u2p_dict[utt] = {time: trans}

    for utt, phones in u2p_dict.items():
        p = [value for key, value in sorted(phones.items())]
        ret += f'{utt}\t{" ".join(p)}\n'

    return ret


if __name__ == '__main__':
    main()
