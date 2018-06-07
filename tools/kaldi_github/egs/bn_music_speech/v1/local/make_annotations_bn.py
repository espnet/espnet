#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script creates four files for each HUB4 Broadcast News
# transcript file. The four files are for the music, speech, ad,
# and other transcripts. Each line of the output files define the
# start and end times of the individual events.
#
# This file is meant to be invoked by make_bn.sh.

import sys, re, os

def is_speech(line):
  if "<Segment" in line and "Speaker=" in line:
    return True
  return False

def is_other_type2(line):
  if "Type=Commercial" in line or "Type=Filler" in line or "Type=Local_News" in line:
    return True
  return False

def is_music(line):
  if "Type=Music" in line:
    return True
  return False

def is_other_type1(line):
  if "Type=Other" in line:
    return True
  return False

def extract_speech(line):
  m = re.search('(?<=S_time=)\d+.\d+', line)
  start = float(m.group(0))
  m = re.search('(?<=E_time=)\d+.\d+', line)
  end = float(m.group(0))
  if start > end:
    print "Skipping annotation where end time is before start time:", line
  return start, end

def extract_other_type2(line):
  m = re.search('(?<=S_time=)\d+.\d+', line)
  start = float(m.group(0))
  m = re.search('(?<=E_time=)\d+.\d+', line)
  end = float(m.group(0))
  if start > end:
    print "Skipping annotation where end time is before start time:", line
  return start, end

def extract_music(line):
  m = re.search('(?<=Time=)\d+.\d+', line)
  time = float(m.group(0))
  m = re.search('(?<=Level=)\w', line)
  level = m.group(0)
  is_on = False
  if level == "L" or level == "H":
    is_on = True
  elif level == "O":
    is_on = False
  else:
    print "Encountered bad token on line:", line
    sys.exit()
  return time, is_on

def extract_other_type1(line):
  m = re.search('(?<=Time=)\d+.\d+', line)
  time = float(m.group(0))
  m = re.search('(?<=Level=)\w', line)
  level = m.group(0)
  is_on = False
  if level == "L" or level == "H":
    is_on = True
  elif level == "O":
    is_on = False
  else:
    print "Encountered bad token on line:", line
    sys.exit()
  return time, is_on

def process_file(annos):
  speech = ""
  music = ""
  other_type2 = ""
  other_type1 = ""
  start_new_music_segment = True
  start_new_other_segment = True
  max_time = 0.0
  prev_music_time = "0.0"
  prev_other_time = "0.0"
  for line in annos:
    if is_speech(line):
      speech_start, speech_end = extract_speech(line)
      speech = speech + str(speech_start) + " " + str(speech_end) + "\n"
      max_time = max(speech_end, max_time)
    elif is_other_type2(line):
      other_type2_start, other_type2_end = extract_other_type2(line)
      other_type2 = other_type2 + str(other_type2_start) + " " + str(other_type2_end) + "\n"
      max_time = max(other_type2_end, max_time)
    elif is_music(line):
      time, is_on = extract_music(line)
      max_time = max(time, max_time)
      if is_on and start_new_music_segment:
        prev_music_time = time
        start_new_music_segment = False
      elif not is_on and not start_new_music_segment:
        music = music + str(prev_music_time) + " " + str(time) + "\n"
        start_new_music_segment = True
    elif is_other_type1(line):
      time, is_on = extract_other_type1(line)
      max_time = max(time, max_time)
      if is_on and start_new_other_segment:
        prev_other_time = time
        start_new_other_segment = False
      elif not is_on and not start_new_other_segment:
        other_type1 = other_type1 + str(prev_other_time) + " " + str(time) + "\n"
        start_new_other_segment = True

  if not start_new_music_segment:
    music = music + str(prev_music_time) + " " + str(max_time) + "\n"
  if not start_new_other_segment:
    other_type1 = other_type1 + str(prev_other_time) + " " + str(max_time) + "\n"

  other = other_type1 + other_type2
  return speech, music, other

def main():
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]
  utts = ""
  for root, dirs, files in os.walk(in_dir):
    for file in files:
      if file.endswith(".txt"):
        anno_in = open(os.path.join(root, file), 'r').readlines()
        speech, music, other = process_file(anno_in)
        utt = file.replace(".txt", "")
        utts = utts + utt + "\n"
        speech_fi_str = utt + "_speech.key"
        music_fi_str = utt +  "_music.key"
        other_fi_str = utt +  "_other.key"
        speech_fi = open(os.path.join(out_dir, speech_fi_str), 'w')
        speech_fi.write(speech)
        music_fi = open(os.path.join(out_dir, music_fi_str), 'w')
        music_fi.write(music)
        other_fi = open(os.path.join(out_dir, other_fi_str), 'w')
        other_fi.write(other)
  utts_fi = open(os.path.join(out_dir, "utt_list"), 'w')
  utts_fi.write(utts)

if __name__=="__main__":
  main()

