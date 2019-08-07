import os
import sys

def write_lab(out_dir, rfile, data):
    os.makedirs(out_dir, exist_ok=True)
    wfile = os.path.join(out_dir, os.path.basename(rfile))
    with open(wfile, "w") as wf:
        for data_row in range(len(data)):
            wf.write(data[data_row][0] + "\t")
            wf.write(data[data_row][1] + "\t")
            wf.write(data[data_row][2] + "\n")

def main():
    args = sys.argv
    list_file = args[1]
    txt_file = args[2]
    out_dir = args[3]

    delimiter = " "
    with open(list_file, "r") as f:
        flist = list(map(lambda x: x.split(delimiter), f.read().strip().split("\n")))

    with open(txt_file, "r") as f:
        new_lab = f.read().strip().split("\n")

    rfile = ""
    for n in range(len(new_lab)):
        row_num = int(flist[n][1]) - 1

        if (rfile != flist[n][0]):
            if (rfile != ""):
                write_lab(out_dir, rfile, data)
        
            rfile = flist[n][0]
            delimiter = "\t"
            with open(rfile, "r") as rf:
                data = list(map(lambda x: x.split(delimiter), rf.read().strip().split("\n")))

        data[row_num][2] = new_lab[n]
    write_lab(out_dir, rfile, data)

if __name__ == "__main__":
    main()
