import os
import sys


def main(srcdir, savedir, audiodir):
    my_file = open(srcdir, "r")
    content_list = my_file.readlines()
    savetexts = []
    for i in range(len(content_list)):
        text = content_list[i].split(" ")[0]
        savetexts.append(" ".join([text, os.path.join(audiodir, text + ".wav")]) + "\n")
    with open(savedir, "a") as textprocess:
        textprocess.writelines(savetexts)
        textprocess.close()


# hand over parameter overview
# sys.argv[1] = srcdir (str), Directory of the old wav.scp file
# sys.argv[2] = savedir(str), Directory to save the new wav.scp file
# sys.argv[3] = audiodir(str), Directory of the audio files
main(sys.argv[1], sys.argv[2], sys.argv[3])
