text = 'text'
out = 'text_cleaned'

f = open(text)
of = open(out, 'w')
for line in f:
    tokens = line.replace('\n', '').split()
    utt = tokens[0]
    phones = tokens[1:]

    out_phones = []
    for p in phones:
        p = ''.join([i for i in p if not i.isdigit()])
        p = p.split('_')[0]
        out_phones.append(p)

    s = f"{utt}\t{' '.join(out_phones)}\n"
    of.write(s)

of.close()
