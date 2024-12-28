#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
[ -z "${HF_TOKEN}" ] && \
    echo "ERROR: You need to setup the variable HF_TOKEN with your HuggingFace access token" && \
exit 1

python app.py \
  --eval_options "Latency,TTS Intelligibility,TTS Speech Quality,ASR WER,Text Dialog Metrics" \
  --tts_options "kan-bayashi/ljspeech_vits,kan-bayashi/libritts_xvector_vits,kan-bayashi/vctk_multi_spk_vits,ChatTTS" \
  --llm_options "meta-llama/Llama-3.2-1B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --asr_options "pyf98/owsm_ctc_v3.1_1B,espnet/owsm_ctc_v3.2_ft_1B,espnet/owsm_v3.1_ebf,librispeech_asr,whisper-large"
