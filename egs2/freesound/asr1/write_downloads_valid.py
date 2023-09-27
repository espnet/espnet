file = open("data/test/wav.scp")
line_arr = [line for line in file]
file_write = open("write_downloads_test1.sh", "w")
for line in line_arr:
    if len(line.split("trim")) > 2:
        line1 = (
            " ".join(line.split()[1:]).split('"')[-2][2:].split(" trim")[0]
            + ' "test_downloads/'
            + line.split()[0]
            + '.wav" trim'
            + " ".join(line.split()[1:]).split('"')[-2][2:].split(" trim")[-1]
        )
        line2 = (
            'sox "/'
            + "/".join(line1.split("/")[1:]).split(".wav")[0]
            + '.wav"'
            + ".wav".join("/".join(line1.split("/")[1:]).split(".wav")[1:])
        )
        file_write.write(line2 + "\n")
