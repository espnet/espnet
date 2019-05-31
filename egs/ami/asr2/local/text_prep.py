import argparse
import codecs
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser("convert .rttm to text kaldi")
    parser.add_argument('rttm', help='rttm used in diarization')
    parser.add_argument('output', help='write the resegmented output')

    args = parser.parse_args()

    segs = codecs.open(args.rttm, 'r', encoding='ascii')
    out = codecs.open(args.output, 'w', encoding='ascii')

    count_min = 5
    count_max = 10
    sample = random.randrange(count_min, count_max)

    spkrs = []
    ids = []
    start = []
    end = []
    i = 0
    for line in segs:
        # extracting necessary segments
        segs = line.split(' ')
        path = segs[1].split('/')[0]
        stime = float(segs[3])
        ttime = float(segs[4])
        etime = stime + ttime
        start.append(stime)
        end.append(etime)
        spkrs.append(segs[7] + '_' + path)
        ids.append(path)
    num = len(start)
    new_spkrs = []
    new_ids = []
    new_start = []
    new_end = []
    new_output = []
    for idx in range(0, num, sample):
        if idx < num-sample:
            new_output.append(ids[0])
            new_output.append(' ')
            new_output.append(start[idx:idx+sample][0])
            new_output.append(' ')
            new_output.append("%.2f" % end[idx + sample])
            new_output.append(' ')
            new_output.append(' '.join(str(spk) for spk in spkrs[idx:idx + sample]))
            new_output.append('\n')
    out.write(''.join([str(line) for line in new_output]))
