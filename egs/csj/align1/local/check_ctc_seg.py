import os
import pdb
import argparse


def time_stamp_to_hms(time_stamp):
    """
    input: <timestamp> e.g. 3456.789
    output: <str> e.g. 00:57:36,789
    """
    hour = int(time_stamp // (60 * 60))
    minute = int((time_stamp - hour * 60 * 60) // 60)
    second = int((time_stamp - hour * 60 * 60 - minute * 60))
    hour_str = "%02d" % hour
    minute_str = "%02d" % minute
    second_str = "%02d" % second
    milisecond = str(time_stamp).split('.')[1]
    return(hour_str + ':' + minute_str + ':' + second_str + ',' + milisecond)


def generate_srt(utt_container, ctc_container, refine=False):
    result = []
    count = 1
    for l in ctc_container:
        utt_id, ses_id, start, end, _ = l.split(' ')
        start = float(start)
        end = float(end)
        if refine:
            end = end + 0.1
        start = time_stamp_to_hms(start)
        end = time_stamp_to_hms(end)
        assert utt_id in utt_container.keys()
        result.append("%d\n%s --> %s\n%s\n" %
                      (count, start, end, utt_container[utt_id]))
        count += 1
    # pdb.set_trace()
    return result


if __name__ == '__main__':
    '''
    Usage: 
    work_dir: Directory to be used
    output: <ses_id>.srt
    '''
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", type=str, default='data/iis20160531')
    p.add_argument("--refine_duration", type=float, default=0.1,
                   help="extend endding of each utt with <refine_duration> second")

    args = p.parse_args()

    work_dir = args.work_dir
    assert os.path.exists(work_dir)
    ses_id = work_dir.split('/')[-1]
    wav_scp = os.path.join(work_dir, 'wav.scp')
    ctc_seg = os.path.join(work_dir, 'aligned_segments_clean')
    utt = os.path.join(work_dir, 'utt_text')

    assert os.path.exists(wav_scp)
    with open(wav_scp, 'r') as f:
        wav_path = f.readline()

    ctc_cnt = []
    assert os.path.exists(ctc_seg)
    with open(ctc_seg, 'r') as f:
        for line in f.readlines():
            ctc_cnt.append(line)

    utt_ids = []
    utt_texts = []
    assert os.path.exists(utt)
    with open(utt, 'r') as f:
        for line in f.readlines():
            utt_ids.append(line.split(' ')[0])
            utt_texts.append(line.split(' ')[1])
    # utt_cnt into dict <id>: <text>
    utt_cnt = dict(zip(utt_ids, utt_texts))

    srt_container = generate_srt(utt_cnt, ctc_cnt, refine=args.refine_duration)
    output_srt = os.path.join(work_dir, ses_id + '.srt')

    with open(output_srt, 'w') as f:
        print(''.join(x for x in srt_container), file=f)
