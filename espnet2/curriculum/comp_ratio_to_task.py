import sys

################ AUX FUNCS ################
def read_CR(cr_file):
    '''
    Reads the input comp_ratio.txt
    The input format: uttID, CR
        10001_8844_000000  [0.4956896]
    Params:
        cr_file (str): path to comp_ratio.txt
    '''
    with open(cr_file, 'r') as fo:
        cr_file = fo.read().split(']')[:-1]

    cr_dict = {}

    for i, line in enumerate(cr_file):
        line = line.replace('[', '').split()
        cr_dict[line[0]] = 1 - float(line[1])
    return cr_dict

nTasks = sys.argv[1]
cr_file = sys.argv[2]
task_file = sys.argv[3]

utt2cr = read_CR(cr_file)

cr_sorted = sorted(utt2cr.items(), key=lambda k: k[1])

nPerTask = len(cr_sorted) / int(nTasks)

with open(task_file, 'w') as f:
    i = 0
    task = 0
    for ID, _ in cr_sorted:
        f.write(ID + " " + str(task) + "\n")
        i += 1
        if i % nPerTask == 0:
            task += 1
        
