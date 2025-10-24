import re



file_write=open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_topk_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response_combined_asr/text_new2","w")
src_file=open("dump/raw_codec_ssl_cot_full_utt2spk_librispeech_100/test_fisher_asr_dialogue/index_files/src_text")
src_arr=[line for line in src_file]
count=0
for generated_text in open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_topk_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response_combined_asr/text_new"):
    a1=len(generated_text.split()[1:])
    a2=len(src_arr[count].split()[1:])
    if a1/a2>3.0:
        if "." in generated_text:
            found=True
            while(a1/a2>3.0):
                if "." not in generated_text:
                    generated_text=" ".join(generated_text.split()[:3*a2])
                    found=False
                    break
                generated_text=".".join(generated_text.split(".")[:-1])
                a1=len(generated_text.split()[1:])
            if found:
                generated_text=generated_text+"."
        else:
            generated_text=" ".join(generated_text.split()[:3*a2])

    file_write.write(generated_text.strip()+"\n")
print(generated_text)