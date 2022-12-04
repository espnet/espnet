# Process audio frames for phonemes duration following a similar process as in:
# https://github.com/NATSpeech/NATSpeech/blob/main/utils/audio/align.py

import argparse
import codecs
import glob
import json
import os
import tqdm

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

import numpy as np
import torch

# To Generate Phonemes from words:
# from montreal_forced_aligner.g2p.generator import PyniniValidator
# from montreal_forced_aligner.models import (
#     G2PModel,
#     ModelManager,
# )
# language = "english_us_mfa"
# manager = ModelManager()
# manager.download_model("g2p", language)
# model_path = G2PModel.get_pretrained_path(language)
# g2p = PyniniValidator(g2p_model_path=model_path, num_pronunciations=1, quiet=True)
# g2p.word_list = "my word list".split(" ")
# phones = g2p.generate_pronunciations()

# stfs = torch.stft(
#             values, n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.n_shift,
#             center=True, normalized=False, onesided = True, return_complex = False
#         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--g2p", type=str, required=True, help="G2P type.")
    parser.add_argument("--silence_symbol", type=str, default="<sil>")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--n_shift", type=int, default=256)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--mfa_offset", type=float, default=0.02)
    parser.add_argument("--max_phonemes_word", type=int, default=None)
    parser.add_argument("input_dir", type=str, default="")
    parser.add_argument("output_dir", type=str, default="")
    args = parser.parse_args()

    listfiles = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    phoneme_tokenizer = PhonemeTokenizer(args.g2p)

    SAMPLERATE = args.samplerate
    MFA_OFFSET = args.mfa_offset
    HOP_SIZE = args.n_shift
    MAX_PHONES = args.max_phonemes_word
    
    def getframe(timestamp, _maxtime=None, offset=True):
        if offset:
            timestamp += MFA_OFFSET
        if _maxtime:
            timestamp = min(timestamp, _maxtime) 
        frame = timestamp * SAMPLERATE / HOP_SIZE + 0.5
        return int(frame)

    sil_tkn = args.silence_symbol   
 
    txt_duration = list()
    txt_phonemes = list()
    txt_wrd_durs = list()
    for fn in tqdm.tqdm(listfiles):
        key = os.path.basename(fn).replace(".json", "")
        with codecs.open(fn, "r", encoding="utf-8") as reader:
            data = json.load(reader)
        
        max_time = data["end"]

        # STFT frames for centered
        max_frames = getframe(max_time, offset=False)

        _words = [x for x in data["tiers"]["words"]["entries"]]
        _phns = (x for x in data["tiers"]["phones"]["entries"])

        previous_stop = 0

        phonemes = list()
        duration = list()
        wrd_duration = list()
        for wrd in _words:
            wrd_bgn, wrd_end, wrd = wrd

            if (previous_stop == 0) and (wrd_bgn > 0):
                phn_end = getframe(wrd_bgn, max_time)
                phonemes.append(sil_tkn)
                duration.append(phn_end)
                wrd_duration.append(1)
            elif len(phonemes) > 0:
                phonemes.append("<space>")
                phn_bgn = getframe(previous_stop, max_time)
                phn_end = getframe(wrd_bgn, max_time)
                duration.append(max(phn_end - phn_bgn, 0))
                wrd_duration.append(1)
            
            align_phn = wrd_bgn
            phns_wrd = list()
            for phn in _phns:
                phn_bgn, phn_end, phn = phn
                if align_phn < phn_bgn:
                    # Add silence if phone starts after previous phone timestamp
                    _phn_bgn = getframe(align_phn, max_time)
                    _phn_end = getframe(phn_bgn, max_time)
                    duration.append(max(_phn_end - _phn_bgn, 0))
                    phns_wrd.append(sil_tkn)   

                align_phn = phn_end
                _phn_bgn = getframe(phn_bgn, max_time, (phn_bgn > 0))
                _phn_end = getframe(phn_end, max_time)
                frames = max(_phn_end - _phn_bgn, 0)

                if phn == "spn":
                    # In case of spn (OOV) the number of frames will be equally 
                    # split between the phones obtained from the phonemizer
                    phn = phoneme_tokenizer.text2tokens(wrd)
                    div = len(phn)
                    frames = [frames // div + (1 if x < frames % div else 0)  for x in range (div)]
                    phns_wrd.extend(phn)
                    duration.extend(frames)
                else:
                    phns_wrd.append(phn)                
                    duration.append(frames)

                if phn_end == wrd_end:
                    break
            previous_stop = wrd_end
            phonemes.extend(phns_wrd)
            _wrd_phn = len(phns_wrd)
            if MAX_PHONES:
                groups, residual = _wrd_phn // MAX_PHONES, _wrd_phn % MAX_PHONES
                _groups = [MAX_PHONES] * groups + ([residual] if residual else [])
                wrd_duration.extend(_groups)
            else:
                wrd_duration.append(_wrd_phn)
        
        final_frm = max_frames - getframe(previous_stop, max_time) - 1
        if final_frm > 0:
            phonemes.append(sil_tkn)
            duration.append(final_frm)
            wrd_duration.append(1)
        elif final_frm < 0:
            duration[-1] -= 1
        
        # insert EOS, duration and word duration should be +1 larger than the phonemes
        duration.append(1)
        wrd_duration.append(1)

        assert (sum(wrd_duration) - 1) == len(phonemes), f"{sum(wrd_duration)} != {len(phonemes)}"
        assert len(phonemes) == (len(duration) - 1), f"{len(phonemes)} != {len(duration)}"
        assert sum(duration) == max_frames, f"{sum(duration)} != {max_frames}"

        phonemes = " ".join(phonemes)
        duration = " ".join(map(str, duration))
        assert len(phonemes.split(" ")) == (len(duration.split(" ")) - 1), f"{len(phonemes)} != {len(duration)}"
        wrd_durs = " ".join(map(str, wrd_duration))
        txt_phonemes.append(f"{key} {phonemes}")
        txt_duration.append(f"{key} {duration}")
        txt_wrd_durs.append(f"{key} {wrd_durs}")
    
    with codecs.open(os.path.join(args.output_dir, "text.phn"), "w", encoding="utf-8") as writer:
        writer.write("\n".join(txt_phonemes))
        
    with codecs.open(os.path.join(args.output_dir, "durations"), "w", encoding="utf-8") as writer:
        writer.write("\n".join(txt_duration))
    
    with codecs.open(os.path.join(args.output_dir, "word_durations"), "w", encoding="utf-8") as writer:
        writer.write("\n".join(txt_wrd_durs))   
