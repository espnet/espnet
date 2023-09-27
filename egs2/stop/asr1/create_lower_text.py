for sets in ["train_sp", "valid", "test"]:
    file_read = open(
        "/projects/bbjs/arora1/espnet/egs2/stop/slu1/dump/raw/" + sets + "/text"
    )
    line_arr = [line for line in file_read]
    line_arr1 = [
        line.split()[0] + " " + " ".join(line.split()[1:]).lower() for line in line_arr
    ]
    file_write = open("dump/raw/" + sets + "/text", "w")
    for line in line_arr1:
        file_write.write(line + "\n")
