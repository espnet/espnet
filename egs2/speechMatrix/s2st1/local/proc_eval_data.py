import os
import argparse

def prep(filename, datadir, src_lang, tgt_lang, utt2spk_fp, src_wav_fp, tgt_text_fp=None, vp=False):
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            # id  src_audio   src_n_frames    tgt_audio   tgt_n_frames
            _id, src, _, tgt, _ = line.split('\t')
            _id = _id.split('/')[-1]
            if vp: 
                # en/es/2010/20100705-0900-PLENARY-16-es_20100705-22:08:00_1
                src = f"{datadir}/voxpopuli_valid_test/{_id}.ogg"
            # Use ffmpeg to convert audio to WAV
            src_wav_fp.write(f"{_id} ffmpeg -i {src} -acodec pcm_s16le -ac 1 -ar 16000 -f wav - |\n")
            utt2spk_fp.write(f"{_id} {src_lang}_{tgt_lang}\n")
    
            if tgt_text_fp:
                tgt_text_fp.write(f"{_id} {tgt}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/ocean/projects/cis210027p/shared/corpora/speechmatrix")
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
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

    tgt_text_fp = None
    if args.subset != "test":
        src_text = f"{args.dest}/text.{args.src_lang}"
        src_text_fp = open(src_text, "w")
        tgt_text = f"{args.dest}/text.{args.tgt_lang}"
        tgt_text_fp = open(tgt_text, "w")
    
    # Fleurs Valid + Test
    split = "valid" if args.subset == "dev" else args.subset
    fleurs = f"{real_dd}/flores200_{split}/s2u_manifests/es-en/{split}_fleurs.tsv"
    prep(fleurs, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp, tgt_text_fp)

    # voxpopuli valid
    if args.subset == "dev":
        voxpopuli = f"{real_dd}/s2u_manifests/es-en/valid_vp.tsv"
        prep(voxpopuli, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp, tgt_text_fp, vp=True)

        src_unit = f"{real_dd}/s2u_manifests/en-es/source_unit/valid_vp.tsv"
        with open(src_unit) as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                _id, src_txt = lines[i].split('\t')
                _id = _id.split('/')[-1]
                src_text_fp.write(f"{_id} {src_txt}")

    # EuroParl Test
    if args.subset == "test":
        europarl = f"{real_dd}/europarl-st-1.1/mined_data/s2u_manifests/es-en/test_epst.tsv"
        prep(europarl, real_dd, src_lang, tgt_lang, utt2spk_fp, wavscp_fp)

    # close files
    wavscp_fp.close()
    utt2spk_fp.close()
    if args.subset != "test":
        src_text_fp.close()
        tgt_text_fp.close()
