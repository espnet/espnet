for sets in ["dev_clean","test_clean","test_other"]:
    file1=open("exp/tts_tts_16k_char_xvector_unpaired_new_gumbel6/decode_train.loss.ave/"+sets+"/text")
    line_arr=[line.replace("<sos/eos>","") for line in file1]
    file2=open("exp/tts_tts_16k_char_xvector_unpaired_new_gumbel6/decode_train.loss.ave/"+sets+"/text_new","w")
    for line in line_arr:
        file2.write(line)
