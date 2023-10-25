import os
import argparse

def prep(filename, datadir, src_wav_fp, tgt_text_fp=None, vp=False):
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
            src_wav_fp.write(f"{_id} {src}\n")
 
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

    # UNIT (text) and PATH (wav.scp)
    wavscp = f"{args.dest}/{args.subset}_{args.src_lang}/wav.scp.{args.src_lang}"
    os.makedirs(os.path.dirname(wavscp), exist_ok=True)
    wavscp_fp = open(wavscp, 'w')

    tgt_text_fp = None
    if args.subset != "test":
        src_text = f"{args.dest}/{args.subset}_{args.src_lang}/text.{args.src_lang}"
        src_text_fp = open(src_text, "w")
        tgt_text = f"{args.dest}/{args.subset}_{args.src_lang}/text.{args.tgt_lang}"
        tgt_text_fp = open(tgt_text, "w")
    
    # Fleurs Valid + Test
    split = "valid" if args.subset == "dev" else args.subset
    fleurs = f"{args.datadir}/flores200_{split}/s2u_manifests/es-en/{split}_fleurs.tsv"
    prep(fleurs, args.datadir, wavscp_fp, tgt_text_fp)

    # voxpopuli valid
    if args.subset == "dev":
        voxpopuli = f"{args.datadir}/s2u_manifests/es-en/valid_vp.tsv"
        prep(voxpopuli, args.datadir, wavscp_fp, tgt_text_fp, vp=True)

        src_unit = f"{args.datadir}/s2u_manifests/en-es/source_unit/valid_vp.tsv"
        with open(src_unit) as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                _id, src_txt = lines[i].split('\t')
                _id = _id.split('/')[-1]
                src_text_fp.write(f"{_id} {src_txt}")

    # EuroParl Test
    if args.subset == "test":
        europarl = f"{args.datadir}/europarl-st-1.1/mined_data/s2u_manifests/es-en/test_epst.tsv"
        prep(europarl, args.datadir, wavscp_fp)

    wavscp_fp.close()
    if args.subset != "test":
        src_text_fp.close()
        tgt_text_fp.close()
