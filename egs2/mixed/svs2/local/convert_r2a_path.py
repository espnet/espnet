import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change relative path to absolute path")
    parser.add_argument("org_base", type=str, help="source data directory")
    parser.add_argument("org_info", type=str, help="original resources file")
    parser.add_argument("tgt_info", type=str, help="target resources file")
    args = parser.parse_args()

    org_file = open(args.org_info, "r", encoding="utf-8")
    tgt_file = open(args.tgt_info, "w", encoding="utf-8")
    for line in org_file.readlines():
        line = line.strip().split()
        uid = line[0]
        directory = " ".join(line[1:])
        tgt_path = os.path.join(args.org_base, directory)
        if not os.path.exists(tgt_path):
            raise FileExistsError(f"{tgt_path} does not exist.")
        tgt_file.write("{} {}\n".format(uid, os.path.join(args.org_base, directory)))
    
    org_file.close()
    tgt_file.close()
