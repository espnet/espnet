replace_dict = {}
replace_dict["em:neu"] = "<neu>"
replace_dict["em:ang"] = "<ang>"
replace_dict["em:sad"] = "<sad>"
replace_dict["em:hap"] = "<hap>"
file = open(
    "exp/"
    + "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/"
    + "decode_asr_er2_asr_model_valid.acc.ave.copy/test_iemocap/text"
)
line_arr = [line for line in file]
line_arr = [
    line.replace("<|er|>", "").replace("<|iemocap|>", "").replace(" SEP", "")
    for line in line_arr
]
line1_arr = []
for line in line_arr:
    line1 = line
    for k in replace_dict:
        if k in line:
            line1 = line.replace(k, replace_dict[k])
    line1_arr.append(line1)
file_write = open(
    "exp/"
    + "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/"
    + "decode_asr_er2_asr_model_valid.acc.ave.copy/test_iemocap/text",
    "w",
)
for line in line1_arr:
    file_write.write(line)
