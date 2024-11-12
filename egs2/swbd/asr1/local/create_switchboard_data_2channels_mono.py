dir_arr = [
    "Train_Two_Channel_Label",
    "Val_Two_Channel_Label",
    "Test_Two_Channel_Label",
]
for x in dir_arr:
    file_write = open(x + "_Mono.csv", "w")
    file = open(x + ".csv")
    line_arr = [line for line in file]
    prev_speaker = "NA"
    prev_file_id = "NA"
    label_arr_write = []
    prev_speaker_arr = []
    exception_count = 0
    exception_count1 = 0
    for i in range(1, len(line_arr)):
        line_arr1 = line_arr[i].strip().split(",")
        if line_arr1[0] != prev_file_id:
            prev_speaker = "NA"
        prev_file_id = line_arr1[0]
        prev_speaker_arr.append(prev_speaker)
        if (line_arr1[-2] == "NA") and (line_arr1[-1] == "NA"):
            label_arr_write.append("NA")
        elif (line_arr1[-2] == "IPU") and (line_arr1[-1] == "NA"):
            if prev_speaker == "A":
                label_arr_write.append("C")
            else:
                if prev_speaker == "AB":
                    label_arr_write.append("C")
                else:
                    label_arr_write.append("T")
                prev_speaker = "A"
        elif (line_arr1[-2] == "NA") and (line_arr1[-1] == "IPU"):
            if prev_speaker == "B":
                label_arr_write.append("C")
            else:
                if prev_speaker == "BA":
                    label_arr_write.append("C")
                else:
                    label_arr_write.append("T")
                prev_speaker = "B"
        elif (line_arr1[-2] == "IPU") and (line_arr1[-1] == "IPU"):
            if prev_speaker == "AB" or prev_speaker == "BA":
                label_arr_write.append("I")
            else:
                label_arr_write.append("I")
                if prev_speaker == "A":
                    prev_speaker = "AB"
                else:
                    prev_speaker = "BA"
        elif (line_arr1[-2] == "BC") and (line_arr1[-1] == "BC"):
            label_arr_write.append("BC_2")
            exception_count += 1
        elif line_arr1[-2] == "BC":
            if prev_speaker != "B" and prev_speaker != "BA":
                exception_count1 += 1
                label_arr_write.append("BC_1")
            else:
                label_arr_write.append("BC")
            # prev_speaker=""
        elif line_arr1[-1] == "BC":
            if prev_speaker != "A" and prev_speaker != "AB":
                exception_count1 += 1
                label_arr_write.append("BC_1")
            else:
                label_arr_write.append("BC")
        else:
            print("Error")
            import pdb

            pdb.set_trace()

    line_arr = line_arr[1:]
    assert len(label_arr_write) == len(line_arr)
    assert len(prev_speaker_arr) == len(line_arr)
    file_write.write(line_arr[0])
    for i in range(len(line_arr)):
        line_arr1 = line_arr[i].strip().split(",")
        file_write.write(
            ",".join(line_arr1[:-2] + [label_arr_write[i], prev_speaker_arr[i]]) + "\n"
        )

    print(exception_count)
    print(exception_count1)
