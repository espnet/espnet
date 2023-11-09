import os
import sys

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    joint_key = lines[0].strip().split(" ")[0] + "*" + lines[1].strip().split(" ")[0]
    with open(os.path.join(sys.argv[2], "trial.scp"), "w") as f:
        f.write(joint_key + " " + " ".join(lines[0].strip().split(" ")[1:]) + "\n")
    with open(os.path.join(sys.argv[2], "trial2.scp"), "w") as f:
        f.write(joint_key + " " + " ".join(lines[1].strip().split(" ")[1:]) + "\n")
    with open(os.path.join(sys.argv[2], "trial_label"), "w") as f:
        f.write(joint_key + " 0\n")
