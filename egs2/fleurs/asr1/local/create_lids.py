with open('dump/raw/train_all_sp/text', 'r', encoding='utf-8') as in_file, open('dump/raw/train_all_sp/lid_utt', 'w', encoding='utf-8') as utt_file, open('dump/raw/train_all_sp/lid_tok', 'w', encoding='utf-8') as tok_file:
    lines = in_file.readlines()
    for line in lines:
        utt_id = line.split()[0]
        lid = line[line.index('['):line.index(']')+1]
        utt_file.write(f"{utt_id} {lid} \n")

        words = line[line.index(']')+1:]
        lids = [lid for word in words.split()]
        tok_file.write(f"{utt_id} {lid} {' '.join(lids)}\n")