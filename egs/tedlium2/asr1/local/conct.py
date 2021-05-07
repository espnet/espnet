import sys

txt = {}
before = 0
after = 0
for line in sys.stdin:
    before  += 1
    splited = line.strip().split()
    uid, trs = splited[0], splited[1:]                                                                                                                                                                          
    sid = uid.split("-")[0]
    trs = " ".join(trs)
    if sid in txt.keys():
        txt[sid].append(" " + trs)
    else:
        txt[sid] = [trs]

for sid in txt.keys():
    after += 1
    print(sid + " ", end="")
    for trs in txt[sid]:
        print(trs, end="")
    print("")
