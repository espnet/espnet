token_file = open("data/en_token_list/whisper_multilingual/tokens.txt")
token_list = [line.strip() for line in token_file]
file = open(
    "exp/"
    + "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/"
    + "decode_asr_lid_asr_model_valid.acc.ave/test_voxforge/token_int"
)
file_write = open(
    "exp/"
    + "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/"
    + "decode_asr_lid_asr_model_valid.acc.ave/test_voxforge/text",
    "w",
)
line_arr = [line for line in file]
result_dict = {}

for line in line_arr:
    file_write.write(line.split()[0] + " " + token_list[int(line.split()[1])] + "\n")
