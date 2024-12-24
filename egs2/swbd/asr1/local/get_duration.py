import json

path_file = open("data/local/train/sph.scp")
file_write = open("sox_duration.sh", "w")
for k in path_file:
    file_write.write("sox " + k.strip().split()[-1] + " -n stat\n")
