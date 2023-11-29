import os
import argparse

def prep(
    filename, 
    datadir, 
    src_lang, 
    tgt_lang, 
    utt2spk_fp, 
    src_wav_fp, 
    src_text_fp=None,
    tgt_text_fp=None,
    option="",
    split="valid",
):
    natural_texts = []
    if split == "test":
        # natural_text
        if option == "europarl":
            natural_text = f"{datadir}/europarl-st-1.1/mined_data/s2u_manifests/es-en/test_epst.en"
        elif option == "fleurs" and split == "test":
            natural_text = f"{datadir}/flores200_mined/s2u_manifests/es-en/test_fleurs.en"
        with open(natural_text) as f:
            natural_texts = f.readlines()
        
        
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            # id  src_audio   src_n_frames    tgt_audio   tgt_n_frames
            _id, src, _, tgt, _ = line.split('\t')
            _id = _id.split('/')[-1]

            if option == "voxpopuli":
                src = f"{datadir}/voxpopuli_valid_test/{_id}.ogg"        
            # Use ffmpeg to convert audio to WAV
            src_wav_fp.write(f"{_id} ffmpeg -i {src} -acodec pcm_s16le -ac 1 -ar 16000 -f wav - |\n")
            utt2spk_fp.write(f"{_id} {src_lang}_{tgt_lang}\n")
    
            if tgt_text_fp:
                if split == "test":
                    tgt = natural_texts[i-1].strip('\n')
                tgt_text_fp.write(f"{_id} {tgt}\n")


    if src_text_fp:
        #NOTE: test set does not have src_unit, so the path is for valid set
        if option == "voxpopuli": 
            # en/es/2010/20100705-0900-PLENARY-16-es_20100705-22:08:00_1
            src = f"{datadir}/voxpopuli_valid_test/{_id}.ogg"
            src_unit = f"{datadir}/s2u_manifests/{src_lang}-{tgt_lang}/source_unit/valid_vp.tsv"
        elif option == "fleurs":
            src_unit = f"{datadir}/flores200_valid/s2u_manifests/{src_lang}-{tgt_lang}/source_unit/valid_fleurs.tsv"

        with open(src_unit) as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                _id, src_txt = lines[i].split('\t')
                _id = _id.split('/')[-1]
                src_text_fp.write(f"{_id} {src_txt}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/ocean/projects/cis210027p/shared/corpora/speechmatrix")
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=["dev", "test"])
    parser.add_argument("--src_lang", type=str, default="es")
    parser.add_argument("--tgt_lang", type=str, default="en")

    args = parser.parse_args()
    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    real_dd = os.path.realpath(args.datadir)

    # UNIT (text) and PATH (wav.scp)
    wavscp = f"{args.dest}/wav.scp.{args.src_lang}"
    os.makedirs(os.path.dirname(wavscp), exist_ok=True)
    wavscp_fp = open(wavscp, 'w')
    utt2spk_fp = open(f"{args.dest}/utt2spk", "w")

    src_text_fp, tgt_text_fp = None, None
    tgt_text = f"{args.dest}/text.{args.tgt_lang}"
    src_text_fp = None
    tgt_text_fp = open(tgt_text, "w")
    if args.subset != "test":
        src_text = f"{args.dest}/text.{args.src_lang}"
        src_text_fp = open(src_text, "w")
    
    # Fleurs Valid + Test
    split = "valid" if args.subset == "dev" else args.subset
    fleurs = f"{real_dd}/flores200_{split}/s2u_manifests/es-en/{split}_fleurs.tsv"
    prep(fleurs, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp, src_text_fp, tgt_text_fp, "fleurs", split)

    # voxpopuli valid
    if args.subset == "dev":
        voxpopuli = f"{real_dd}/s2u_manifests/es-en/valid_vp.tsv"
        prep(voxpopuli, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp, src_text_fp, tgt_text_fp, "voxpopuli")

    # EuroParl Test
    if args.subset == "test":
        europarl = f"{real_dd}/europarl-st-1.1/mined_data/s2u_manifests/es-en/test_epst.tsv"
        prep(europarl, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp, src_text_fp, tgt_text_fp, "europarl", "test")

    # close files
    wavscp_fp.close()
    utt2spk_fp.close()
    tgt_text_fp.close()
    if args.subset != "test":
        src_text_fp.close()
