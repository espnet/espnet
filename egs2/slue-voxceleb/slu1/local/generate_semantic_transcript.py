for dset in ["train_sp"]:
	file1=open("../asr1/exp/asr_train_asr_conformer_lr2e-3_warmup5k_conv2d_seed1999_raw_en_word_sp/decode_asr_yifan_asr_model_valid.acc.ave_10best/train_sp/text")
	file2=open("dump/raw/"+dset+"/transcript")
	line1_array=[line1 for line1 in file1]
	line2_array=[line2 for line2 in file2]
	line2_dict={}
	for line in line2_array:
		line2_dict[line.split()[0]]=" ".join(line.split()[1:])
	file3=open("dump/raw/"+dset+"/transcript_new","w")
	for line1 in line1_array:
		if line1.split()[0] in line2_dict:
			if len(line1.split()) == 2:
				text = "<blank>"
			else:
				text = line1.split()[2].replace("▁", "")
			for sub_word in line1.split()[3:]:
				if "▁" in sub_word:
					text = text + " " + sub_word.replace("▁", "")
				else:
					text = text + sub_word
			file3.write(line1.split()[0]+" "+text+"\n")
