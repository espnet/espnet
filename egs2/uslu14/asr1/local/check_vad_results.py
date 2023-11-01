token_file = open("data/en_token_list/whisper_multilingual/tokens.txt")
token_list = [line.strip() for line in token_file]
file = open(
    "exp/"
    + "asr_train_asr_whisper_full_correct_specaug_raw_en_whisper_multilingual/"
    + "decode_asr_vad_asr_model_valid.acc.ave/test_freesound/token_int"
)
line_arr = [line for line in file]
result_dict = {}

for line in line_arr:
    if token_list[int(line.split()[1])] == "<|nospeech|>":
        result_dict[line.split()[0]] = 1
    else:
        result_dict[line.split()[0]] = 0
gt_file = open("dump/raw/test_freesound/text")
gt_line_arr = [line for line in gt_file]
gt_dict = {}

for line in gt_line_arr:
    if line.split()[1] == "vad_class:background":
        gt_dict[line.split()[0]] = 1
    else:
        gt_dict[line.split()[0]] = 0
error = 0
for k in result_dict:
    if gt_dict[k] != result_dict[k]:
        error += 1
print(error / len(result_dict))
