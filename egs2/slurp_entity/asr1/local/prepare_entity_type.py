def combine_type(input_file, output_file):
    for line in input_file:
        str2 = line.split("▁SEP")
        str1 = str2[1:-1]
        type_arr = []
        name_arr = []
        for k in str1:
            print(k)
            ent_type = k.split("▁FILL")[0]
            name = k.split("▁FILL")[1]
            ent_type = ent_type.replace(" ", "").replace("▁", "")
            type_arr.append(ent_type)
            name_arr.append(name)
        final_str = str2[0]
        for k in range(len(type_arr)):
            final_str = final_str + "▁SEP " + type_arr[k] + " ▁FILL" + name_arr[k]
        final_str = final_str + "▁SEP"
        final_str = final_str + str2[-1]
        output_file.write(final_str)


combine_type(open("data/train/text", "r"), open("data/train/text_new", "w"))
combine_type(open("data/devel/text", "r"), open("data/devel/text_new", "w"))
combine_type(open("data/test/text", "r"), open("data/test/text_new", "w"))
