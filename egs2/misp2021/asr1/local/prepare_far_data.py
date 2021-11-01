#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import glob
import codecs
import argparse
from multiprocessing import Pool
import pdb


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r', encoding='utf8') as handle:
            lines_content = handle.readlines()
        processed_lines = list(map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content))
        return processed_lines
    else:
        processed_lines = list(map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content))
        # pdb.set_trace()
        with codecs.open(textpath, 'w', encoding='utf8') as handle:
            handle.write(''.join(processed_lines))
        return None


# class definition
class TextGrid(object):
    def __init__(self, file_type='', object_class='', xmin=0., xmax=0., tiers_status='', tiers=[]):
        self.file_type = file_type
        self.object_class = object_class
        self.xmin = xmin
        self.xmax = xmax
        self.tiers_status = tiers_status
        self.tiers = tiers

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        new_xmax = xend - xstart + self.xmin
        new_xmin = self.xmin
        new_tiers = []

        for tier in self.tiers:
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend))
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=new_xmin, xmax=new_xmax,
                        tiers_status=self.tiers_status, tiers=new_tiers)


class Tier(object):
    def __init__(self, tier_class='', name='', xmin=0., xmax=0., intervals=[]):
        self.tier_class = tier_class
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.intervals = intervals

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        bias = xstart - self.xmin
        new_xmax = xend - bias
        new_xmin = self.xmin
        new_intervals = []
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                new_intervals.append(Interval(xmin=new_xmin, xmax=interval.xmax - bias, text=interval.text))
            elif interval.xmax > xend:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=new_xmax, text=interval.text))
            else:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=interval.xmax - bias, text=interval.text))

        return Tier(tier_class=self.tier_class, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)


class Interval(object):
    def __init__(self, xmin=0., xmax=0., text=''):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))


# io
def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = handle.readlines()
    if lines[-1] == '\r\n':
        lines = lines[:-1]

    assert 'File type' in lines[0], 'error line 0, {}'.format(lines[0])
    file_type = lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'Object class' in lines[1], 'error line 1, {}'.format(lines[1])
    object_class = lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert lines[2] == '\r\n', 'error line 2, {}'.format(lines[2])

    assert 'xmin' in lines[3], 'error line 3, {}'.format(lines[3])
    xmin = float(lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in lines[4], 'error line 4, {}'.format(lines[4])
    xmax = float(lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'tiers?' in lines[5], 'error line 5, {}'.format(lines[5])
    tiers_status = lines[5].split('?')[1].replace(' ', '').replace('\r', '').replace('\n', '')

    assert 'size' in lines[6], 'error line 6, {}'.format(lines[6])
    size = int(lines[6].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert lines[7] == 'item []:\r\n', 'error line 7, {}'.format(lines[7])

    tier_start = []
    for item_idx in range(size):
        tier_start.append(lines.index(' ' * 4 + 'item [{}]:\r\n'.format(item_idx + 1)))

    tier_end = tier_start[1:]+ [len(lines)]

    tiers = []
    for tier_idx in range(size):
        tiers.append(read_tier_from_lines(tier_lines=lines[tier_start[tier_idx] + 1: tier_end[tier_idx]]))

    return TextGrid(file_type=file_type, object_class=object_class, xmin=xmin, xmax=xmax, tiers_status=tiers_status,
                    tiers=tiers)


def read_tier_from_lines(tier_lines):
    assert 'class' in tier_lines[0], 'error line 0, {}'.format(tier_lines[0])
    tier_class = tier_lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'name' in tier_lines[1], 'error line 1, {}'.format(tier_lines[1])
    name = tier_lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'xmin' in tier_lines[2], 'error line 2, {}'.format(tier_lines[2])
    xmin = float(tier_lines[2].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in tier_lines[3], 'error line 3, {}'.format(tier_lines[3])
    xmax = float(tier_lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'intervals: size' in tier_lines[4], 'error line 4, {}'.format(tier_lines[4])
    intervals_num = int(tier_lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert len(tier_lines[5:]) == intervals_num * 5, 'error lines'

    intervals = []
    for intervals_idx in range(intervals_num):
        assert tier_lines[5 + 5 * intervals_idx + 0] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
        assert tier_lines[5 + 5 * intervals_idx + 1] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
        intervals.append(read_interval_from_lines(
            interval_lines=tier_lines[7 + 5 * intervals_idx: 10 + 5 * intervals_idx]))
    return Tier(tier_class=tier_class, name=name, xmin=xmin, xmax=xmax, intervals=intervals)


def read_interval_from_lines(interval_lines):
    assert len(interval_lines) == 3, 'error lines'

    assert 'xmin' in interval_lines[0], 'error line 0, {}'.format(interval_lines[0])
    xmin = float(interval_lines[0].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in interval_lines[1], 'error line 1, {}'.format(interval_lines[1])
    xmax = float(interval_lines[1].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'text' in interval_lines[2], 'error line 2, {}'.format(interval_lines[2])
    text = interval_lines[2].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    return Interval(xmin=xmin, xmax=xmax, text=text)


# wav.scp <recording-id> <extended-filename>
def prepare_wav_scp(data_root, store_dir, wav_root=None, set_type='train'):
    split_rooms = text2lines(textpath=os.path.join(data_root, 'train_dev_test.txt'), lines_content=None)
    room_map = {}
    for line in split_rooms:
        data_type, rooms = line.split(' ')
        room_map[data_type] = rooms.split(',')
    wav_postfix = 'Far.wav'
    id_postfix = 'Far'
    # import pdb
    # # pdb.set_trace()
    all_wav_lines = []
    for room in room_map[set_type]:
        if wav_root:
            wav_list = sorted(glob.glob(os.path.join(wav_root, room, '*', '*', '*_' + wav_postfix)))
        else:
            wav_list = sorted(glob.glob(os.path.join(data_root, room, '*', '*', '*_' + wav_postfix)))
        for wav_path in wav_list:
            id_prefix = os.path.split(wav_path)[-1].split('_')[:4]
            all_wav_lines.append('{}_{} {}'.format('_'.join(id_prefix), id_postfix, wav_path))
    if not os.path.exists('{}/temp'.format(store_dir)):
        os.makedirs('{}/temp'.format(store_dir))
    text2lines(textpath='{}/temp/wav.scp'.format(store_dir), lines_content=all_wav_lines)
    return


# segments <utterance-id> <recording-id> <segment-begin> <segment-end>
# text <utterance-id> <words>
# utt2spk <utterance-id> <speaker-id>
def prepare_segments_text_utt2spk_worker(store_dir, tg_root=None, wav_root=None, processing_id=None,
                                         processing_num=None):
    segments_lines = []
    text_sentence_lines = []
    utt2spk_lines = []
    tier_name = '内容层'
    rejected_text_list = ['<NOISE>', '<DEAF>']
    punctuation_list = ['。', '，', '？']
    sound_list = ['呃', '啊', '噢', '嗯', '唉']
    min_duration = 0.04

    wav_lines = sorted(text2lines(textpath='{}/temp/wav.scp'.format(store_dir), lines_content=None))
    for wav_idx in range(len(wav_lines)):
        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            wav_id, wav_path = wav_lines[wav_idx].split(' ')
            tg_dir = os.path.split(wav_path)[0]
            if wav_root:
                tg_dir = tg_dir.replace(wav_root, tg_root)
            room, speakers, config, index = wav_id.split('_')[:4]
            speaker_list = [speakers[i: i+3] for i in range(1, len(speakers), 3)]
            for speaker in speaker_list:
                tg = read_textgrid_from_file(filepath=os.path.join(
                    tg_dir, '{}_{}_{}_{}_Near_{}.TextGrid'.format(room, speakers, config, index, speaker)))
                target_tier = False
                for tier in tg.tiers:
                    if tier.name == tier_name:
                        target_tier = tier
                if not target_tier:
                    raise ValueError('no tier: {}'.format(tier_name))
                for interval in target_tier.intervals:
                    if interval.text not in rejected_text_list and interval.xmax - interval.xmin >= min_duration:
                        start_stamp = interval.xmin - interval.xmin % 0.04
                        start_stamp = round(start_stamp, 2)
                        end_stamp = interval.xmax + 0.04 - interval.xmax % 0.04 if interval.xmax % 0.04 != 0 else \
                            interval.xmax
                        end_stamp = round(end_stamp, 2)
                        utterance_id = 'S{}_{}_{}_{}_{}_'.format(speaker, room, speakers, config, index) + \
                                       '{0:06d}'.format(round(int(start_stamp*100), 0)) + '-' + \
                                       '{0:06d}'.format(round(int(end_stamp*100), 0))
                        text = interval.text
                        for punctuation in punctuation_list:
                            text = text.replace(punctuation, '')
                        if text not in sound_list:
                            segments_lines.append('{} {} {} {}'.format(utterance_id, wav_id, start_stamp, end_stamp))
                            text_sentence_lines.append('{} {}'.format(utterance_id, text))
                            utt2spk_lines.append('{} S{}'.format(utterance_id, speaker))
    return [segments_lines, text_sentence_lines, utt2spk_lines]


def prepare_segments_text_utt2spk_manager(store_dir, tg_root=None, wav_root=None, processing_num=1):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        all_result = []
        for i in range(processing_num):
            part_result = pool.apply_async(prepare_segments_text_utt2spk_worker, kwds={
                'store_dir': store_dir, 'tg_root': tg_root, 'wav_root': wav_root, 'processing_id': i,
                'processing_num': processing_num})
            all_result.append(part_result)
        pool.close()
        pool.join()
        segments_lines, text_sentence_lines, utt2spk_lines = [], [], []
        for item in all_result:
            part_segments_lines, part_text_sentence_lines, part_utt2spk_lines = item.get()
            segments_lines += part_segments_lines
            text_sentence_lines += part_text_sentence_lines
            utt2spk_lines += part_utt2spk_lines
    else:
        segments_lines, text_sentence_lines, utt2spk_lines = prepare_segments_text_utt2spk_worker(
            store_dir=store_dir, tg_root=tg_root, wav_root=wav_root)

    text2lines(textpath='{}/temp/segments'.format(store_dir), lines_content=segments_lines)
    text2lines(textpath='{}/temp/text_sentence'.format(store_dir), lines_content=text_sentence_lines)
    text2lines(textpath='{}/temp/utt2spk'.format(store_dir), lines_content=utt2spk_lines)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('data_root', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='root directory of dataset')
    parser.add_argument('enhancement_root', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='root directory of enhanced wav')
    parser.add_argument('set_type', type=str, default='train', help='set type')
    parser.add_argument('store_dir', type=str, default='data/train_far', help='set types')
    parser.add_argument('-o', '--only_wav', type=bool, default=False, help='only prepare wav.scp')
    parser.add_argument('-nj', type=int, default=15, help='number of process')
    args = parser.parse_args()

    print('Preparing wav.scp in {} for {} set'.format(args.store_dir, args.set_type))
    prepare_wav_scp(data_root=args.data_root, wav_root=args.enhancement_root, store_dir=args.store_dir,
                    set_type=args.set_type)
    if not args.only_wav:
        print('Preparing segments,text_sentence,utt2spk in {} for {} set'.format(args.store_dir, args.set_type))
        prepare_segments_text_utt2spk_manager(store_dir=args.store_dir, tg_root=args.data_root,
                                              wav_root=args.enhancement_root, processing_num=args.nj)

          