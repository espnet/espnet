#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!"""

import os
import librosa
import argparse

from utils import *


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Extract Audio Waveforms')
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving audio waveforms (.npz)')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

lines = open(args.filename_path).read().splitlines()
lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) if args.testset_only else lines

for filename_idx, line in enumerate(lines):

    filename, person_id = line.split(',')
    print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))
    video_pathname = os.path.join(args.video_direc, filename+'.mp4')
    dst_pathname = os.path.join( args.save_direc, filename+'.npz')

    assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)

    data = librosa.load(video_pathname, sr=16000)[0][-19456:]
    save2npz(dst_pathname, data=data)
