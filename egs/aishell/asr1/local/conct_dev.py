import sys

txt = {}
before = 0
after = 0
for line in sys.stdin:
    before += 1
    splited = line.strip().split()
    uid, trs = splited[0], splited[1:]
    sid = uid.split("W")[0]
    trs = "".join(trs)

    if sid in txt.keys():
        txt[sid].append(trs)
    else:
        txt[sid] = [trs]

txt_2 = {}
for sid in txt.keys():
    txt_2[sid + "A"] = txt[sid][len(txt[sid]) // 2 :]
    txt_2[sid + "B"] = txt[sid][: len(txt[sid]) // 2]

for sid in txt_2.keys():
    after += 1
    print(sid + " ", end="")
    print("".join(txt_2[sid]))
    # for trs in txt[sid]:
    #    print(trs, end="")
