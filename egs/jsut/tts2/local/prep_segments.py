import argparse
import os
from nnmnkwii.io import hts
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        description='Prepare segments from HTS-style alignment files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('wav_scp', type=str, help='wav scp file')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.wav_scp) as f:
        for l in f:
            utt_id, path = l.split()
            lab_path = path.replace("wav/", "lab/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)

            labels = hts.load(lab_path)
            assert "sil" in labels[0][-1]
            assert "sil" in labels[-1][-1]
            segment_begin = "{:.3f}".format(labels[0][1] * 1e-7)
            segment_end = "{:.3f}".format(labels[-1][0] * 1e-7)
            # TODO: make sure this is correct
            # recording_id = "{}_{}_{}".format(utt_id, segment_begin, segment_end)
            sys.stdout.write("{} {} {} {}\n".format(utt_id, utt_id, segment_begin, segment_end))
