import os
import numpy as np
import librosa
from scipy.io import wavfile
from copy import deepcopy

UNK = ['(())']
NONWORD = '~'
EPS = 1e-3
LOGEPS = -30
DEBUG = True

def VAD(y, fs, thres=EPS, merge_thres=0.02, coeff=1.0):
  L = y.shape[0]
  window_len = min(int(fs * 0.1), L)
  dur = float(y.shape[0]/fs)
  exp_filter = coeff ** (np.arange(window_len)[::-1])
  exp_filter /= window_len
  y_smooth = np.convolve(np.abs(y), exp_filter, 'same')
  mask = (y_smooth > thres).astype(float)
  mask_diff = np.diff(mask)

  start_nonsils = np.where(mask_diff > 0)[0] 
  end_nonsils = np.where(mask_diff < 0)[0]
  start_sec_nonsils = [float(start/fs) for start in start_nonsils]
  end_sec_nonsils = [float(end/fs) for end in end_nonsils]
  
  if len(end_sec_nonsils) < len(start_sec_nonsils):
    end_sec_nonsils.append(dur)
  
  segments = [(start, end) for start, end in zip(start_sec_nonsils, end_sec_nonsils)]
  segments_merged = []
  # Merge close segments
  for i_seg, seg in enumerate(segments):
    if i_seg == 0:
      segments_merged.append(seg)
      continue

    prev_merged_seg = segments_merged[-1]
    if seg[0] - prev_merged_seg[1] < 0:
      print('Overlapped segment: end of seg1 %.1f s, start of seg2 %.1f s' % (prev_merged_seg[1], seg[0]))      
      continue
    elif round(seg[0] - prev_merged_seg[1], 2) <= merge_thres:
      segments_merged.pop(-1)
      segments_merged.append((prev_merged_seg[0], seg[1]))
    else:
      # print('prev_merged_seg, seg, prev_merged_seg - seg: ', prev_merged_seg, seg, round(seg[0] - prev_merged_seg[1], 2))
      segments_merged.append(seg)
  
  start_sec_nonsils_merged = [seg[0] for seg in segments_merged]
  end_sec_nonsils_merged = [seg[1] for seg in segments_merged]

  # print('Before merge: ', segments)
  # print('After merge: ', segments_merged)
  return start_sec_nonsils_merged, end_sec_nonsils_merged

def VAD2(y, fs, thres=EPS, merge_thres=0.2):
  L = y.shape[0]
  dur = float(y.shape[0] / fs)
  n_fft = int(25 * fs / 1000)
  hop_length = int(10 * fs / 1000)
  # win_len = 2
  
  sgram = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))**2
  energies = np.sum(sgram, axis=0)
  energies /= max(np.max(energies), EPS)
  # energies -= energies.mean()
  # sgram /= max(np.std(energies), EPS)  

  mask = (energies > thres).astype(float)
  mask_diff = np.diff(np.append(np.zeros((1,)), mask))
  start_nonsils = np.where(mask_diff > 0)[0] 
  end_nonsils = np.where(mask_diff < 0)[0]
  start_sec_nonsils = [round(float(start * hop_length) / fs, 2) for start in start_nonsils]
  end_sec_nonsils = [round(float(end * hop_length) / fs, 2) for end in end_nonsils]
  if len(end_sec_nonsils) < len(start_sec_nonsils):
    end_sec_nonsils.append(dur)

  segments = [(start, end) for start, end in zip(start_sec_nonsils, end_sec_nonsils)]
  segments_merged = []
  # Merge close segments
  for i_seg, seg in enumerate(segments):
    if i_seg == 0:
      segments_merged.append(seg)
      continue

    prev_merged_seg = segments_merged[-1]
    if seg[0] - prev_merged_seg[1] < 0:
      print('Overlapped segment: end of seg1 %.1f s, start of seg2 %.1f s' % (prev_merged_seg[1], seg[0]))      
      continue
    elif round(seg[0] - prev_merged_seg[1], 2) <= merge_thres:
      segments_merged.pop(-1)
      segments_merged.append((prev_merged_seg[0], seg[1]))
    else:
      # print('prev_merged_seg, seg, prev_merged_seg - seg: ', prev_merged_seg, seg, round(seg[0] - prev_merged_seg[1], 2))
      segments_merged.append(seg)
  
  start_sec_nonsils_merged = [seg[0] for seg in segments_merged]
  end_sec_nonsils_merged = [seg[1] for seg in segments_merged]

  # print('Before merge: ', segments)
  # print('After merge: ', segments_merged)
  # XXX
  return start_sec_nonsils_merged, end_sec_nonsils_merged, energies, mask

class BabelKaldiPreparer:
  def __init__(self, data_root, exp_root, sph2pipe, configs):
    self.audio_type = configs.get('audio_type', 'scripted')
    self.is_segment = configs.get('is_segment', False)
    self.create_dev = configs.get('create_dev', True)
    self.nonsilence_interval_file = configs.get('nonsilence_interval_file', 'nonsilence_intervals')
    self.vad = configs.get('vad', True)
    self.verbose = configs.get('verbose', 0)
    self.fs = 8000
    self.data_root = data_root
    self.exp_root = exp_root
    self.sph2pipe = sph2pipe
    self.nonsilence_interval_files = {x: os.path.join('data', x, self.nonsilence_interval_file) for x in ['train', 'test', 'dev']}

    if self.audio_type == 'conversational':
      # XXX Use sub-train
      self.transcripts = {'train': os.listdir(data_root+'conversational/training/transcript_roman/'),
                          'test': os.listdir(data_root+'conversational/eval/transcript_roman/'),
                          'dev':  os.listdir(data_root+'conversational/dev/transcript_roman/')} 
      self.audios = {'train': os.listdir(data_root+'conversational/training/audio/'),
                     'test': os.listdir(data_root+'conversational/eval/audio/'),
                     'dev':  os.listdir(data_root+'conversational/dev/audio/')}  
    elif self.audio_type == 'scripted':
      self.transcripts = {'train': os.listdir(data_root+'scripted/training/transcript_roman/'),
                          'test': os.listdir(data_root+'scripted/training/transcript_roman/'),
                          'dev':  os.listdir(data_root+'scripted/training/transcript_roman/')}  
      self.audios = {'train': os.listdir(data_root+'scripted/training/audio/'),
                     'test': os.listdir(data_root+'scripted/eval/audio/'),
                     'dev':  os.listdir(data_root+'scripted/dev/audio/')} 
    else:
      raise NotImplementedError

  def prepare_tts(self):
    if not os.path.isdir('data'):
      os.mkdir('data')
      os.mkdir('data/train')
      os.mkdir('data/test')
      os.mkdir('data/dev')
    
    if not os.path.isdir(self.exp_root):
      os.mkdir(exp_root)
      os.mkdir(exp_root + 'train')
      os.mkdir(exp_root + 'test')
      os.mkdir(exp_root + 'dev')
   
    if self.audio_type == 'conversational':    
      sph_dir = {
      'train': self.data_root + 'conversational/training/',
      'test': self.data_root + 'conversational/eval/',
      'dev':  self.data_root + 'conversational/dev/'
      }
    elif self.audio_type == 'scripted':
      sph_dir = {
      'train': self.data_root + 'scripted/training/',
      'test': self.data_root + 'scripted/training/',
      'dev':  self.data_root + 'scripted/training/'
      }
    else:
      raise NotImplementedError

    for x in ['train', 'dev', 'test']:
      with open(os.path.join('data', x, 'text'), 'w') as text_f, \
           open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
           open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f, \
           open(os.path.join('data', x, 'segments'), 'w') as segment_f, \
           open(os.path.join('data', x, 'nonsilence_intervals'), 'w') as nonsil_f:
  
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        segment_f.truncate()
 
        i = 0
        for transcript_fn in sorted(self.transcripts[x], key=lambda x:x.split('.')[0]):
          for audio_fn in self.audios[x]:
            if audio_fn.split('.')[0] == transcript_fn.split('.')[0]:
              break 
          # XXX
          if i > 200:
            continue
          i += 1
            
          if self.create_dev:
            if x == 'train' and (i > 0.9 * len(self.transcripts[x])):
              continue
            elif (x == 'test' or x == 'dev') and (i < 0.9 * len(self.transcripts[x])):
              continue  

          utt_id = transcript_fn.split('.')[0]
          # audio_fn = utt_id + '.sph'
          # XXX
          # if utt_id != 'BABEL_BP_101_28204_20111025_133714_inLine': 
          #    continue

          # Load audio
          os.system('%s -f wav -p -c 1 %s temp.wav' % (self.sph2pipe, sph_dir[x] + 'audio/' + audio_fn))

          y, _ = librosa.load('temp.wav', sr=self.fs)  
          sent = []

          if self.is_segment:
            i_seg_all = 0
            print(i, x, utt_id)
            with open(sph_dir[x] + 'transcript_roman/' + transcript_fn, 'r') as transcript_f:
              lines = transcript_f.readlines()
              for i_seg, (start, segment, end) in enumerate(zip(lines[::2], lines[1::2], lines[2::2])):
                i_seg_all += 1
                # XXX
                # if i_seg_all > 20000:
                #   continue

                start_sec = float(start[1:-2])
                end_sec = float(end[1:-2])
                start = int(self.fs * start_sec)
                end = int(self.fs * end_sec)
                if start_sec >= end_sec:
                  if self.verbose > 0:
                    print('Corrupted segment info')
                  continue

                if end_sec - start_sec >= 30:
                  print('Audio sequence too long: ', utt_id, start_sec, end_sec)
                  continue

                words = segment.strip().split(' ')
                words = [w for w in words if w not in UNK and (w[0] != '<' or w[-1] != '>') and w not in NONWORD]
                subsegments = [] 
                sent += words
                
                # Remove utterances that are too long
                if len(words) == 0:
                  if self.verbose > 0:
                    print('Empty segment')
                  continue
                elif len(words) > 400:
                  print('Phone sequence too long: ', utt_id, len(words))
                  continue
                
                # Skip silence interval
                if self.vad:
                  start_sec_nonsils, end_sec_nonsils, energies, mask = VAD2(y[start:end+1], self.fs)
                  if len(start_sec_nonsils) == 0:
                    print('Audio too quiet: ', utt_id, start_sec, end_sec)
                    continue 
                  if end_sec_nonsils[-1] < start_sec_nonsils[0]:
                    print('Corrupted nonsilence interval: ', end_sec_nonsils[-1], start_sec_nonsils[0])
                    print('Energies: ', energies)
                    print('Mask: ', mask)
                    continue

                  segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_sec+start_sec_nonsils[0], start_sec+end_sec_nonsils[-1])) 
                  for i_subseg, (start_sec_nonsil, end_sec_nonsil) in enumerate(zip(start_sec_nonsils, end_sec_nonsils)):
                    nonsil_f.write('%s_%04d_%02d %s %.2f %.2f\n' % (utt_id, i_seg, i_subseg, utt_id, start_sec+start_sec_nonsils[i_subseg], start_sec+end_sec_nonsils[i_subseg]))
                else:
                  segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_sec, end_sec)) 

                text_f.write('%s_%04d %s\n' % (utt_id, i_seg, ' '.join(words)))            
                utt2spk_f.write('%s_%04d %s\n' % (utt_id, i_seg, utt_id)) # XXX dummy speaker id
            if len(sent) == 0:
              continue
            wav_scp_f.write(utt_id + ' ' + self.sph2pipe + ' -f wav -p -c 1 ' + \
                    os.path.join(sph_dir[x], 'audio/', utt_id + '.sph') + ' |\n')
          else:
            print(x, utt_id)
            sent = []
            with open(sph_dir[x] + 'transcript_roman/' + transcript_fn, 'r') as transcript_f:
              lines = transcript_f.readlines()
              for line in lines[1::2]:
                words = line.strip().split(' ')             
                words = [w for w in words if w not in UNK and (w[0] != '<' or w[-1] != '>')]
                sent += words
              
              if len(words) == 0:
                if self.verbose > 0:
                  print('Empty transcript file')
                continue

            text_f.write(utt_id + ' ' + ' '.join(sent) + '\n')
            wav_scp_f.write(utt_id + ' ' + self.sph2pipe + ' -f wav -p -c 1 ' + \
                os.path.join(sph_dir[x], 'audio/', utt_id + '.sph') + ' |\n')
            utt2spk_f.write(utt_id + ' ' + utt_id) # XXX dummy speaker id

      if not self.is_segment:
        os.remove(os.path.join('data', x, 'segments'))
  
  def remove_silence(self, out_dir='BABEL_BP_101_NONSIL/'):
    if not os.path.isdir('data'):
      os.mkdir('data')

    if not os.path.isdir('data/train_nonsil'):
      os.mkdir('data/train_nonsil')
    if not os.path.isdir('data/test_nonsil'):
      os.mkdir('data/test_nonsil')
    if not os.path.isdir('data/dev_nonsil'):
      os.mkdir('data/dev_nonsil')
    
    if not os.path.isdir(out_dir):
      os.mkdir(out_dir)
      os.mkdir(out_dir + 'train')
      os.mkdir(out_dir + 'test')
      os.mkdir(out_dir + 'dev')

    if not os.path.isdir(out_dir + 'train'):
      os.mkdir(out_dir + 'train')
    if not os.path.isdir(out_dir + 'test'):
      os.mkdir(out_dir + 'test')
    if not os.path.isdir(out_dir + 'dev'):
      os.mkdir(out_dir + 'dev')

    if not os.path.isdir(self.exp_root):
      os.mkdir(exp_root)
      os.mkdir(exp_root + 'train_nonsil')
      os.mkdir(exp_root + 'test_nonsil')
      os.mkdir(exp_root + 'dev_nonsil')
 
    if self.audio_type == 'conversational':    
      sph_dir = {
      'train': self.data_root + 'conversational/training/',
      'test': self.data_root + 'conversational/eval/',
      'dev':  self.data_root + 'conversational/dev/'
      }
    elif self.audio_type == 'scripted':
      sph_dir = {
      'train': self.data_root + 'scripted/training/',
      'test': self.data_root + 'scripted/training/',
      'dev':  self.data_root + 'scripted/training/'
      }
    else:
      raise NotImplementedError
    
    nonsilence_intervals = {}

    # XXX
    for x in ['train']: #['train', 'dev', 'test']:
      with open(self.nonsilence_interval_files[x], 'r') as nonsil_f:
        for line in nonsil_f:
          utt_id = line.split()[1]
          seg_id = line.split()[0].split('_')[-2]
          
          if utt_id not in nonsilence_intervals:
            nonsilence_intervals[utt_id] = {}

          start_sec, end_sec = float(line.split()[-2]), float(line.split()[-1])
          if seg_id in nonsilence_intervals[utt_id]:
            nonsilence_intervals[utt_id][seg_id].append([start_sec, end_sec])
          else:
            nonsilence_intervals[utt_id][seg_id] = [[start_sec, end_sec]]
  
    # XXX
    for x in ['train']: #['train', 'dev', 'test']:
      if self.is_segment:
        with open(os.path.join('data', x+'_nonsil', 'text'), 'w') as text_f, \
             open(os.path.join('data', x+'_nonsil', 'wav.scp'), 'w') as wav_scp_f, \
             open(os.path.join('data', x+'_nonsil', 'utt2spk'), 'w') as utt2spk_f, \
             open(os.path.join('data', x+'_nonsil', 'segments'), 'w') as segment_f:  
          text_f.truncate()
          wav_scp_f.truncate()
          utt2spk_f.truncate()
          segment_f.truncate()

          i = 0 
          for transcript_fn in sorted(self.transcripts[x], key=lambda x:x.split('.')[0]):
            utt_id = transcript_fn.split('.')[0]
            if utt_id not in nonsilence_intervals:
              continue

            for audio_fn in self.audios[x]:
              if audio_fn.split('.')[0] == transcript_fn.split('.')[0]:
                break 

            # XXX
            # if i > 2:
            #   continue
            i += 1

            # XXX
            # if utt_id != 'BABEL_BP_101_67798_20111104_013951_inLine': 
            #   print(utt_id)
            #   continue
            print(i, x, utt_id)
            print(audio_fn)
            # Convert .SPH files into .wav
            os.system('%s -f wav -p -c 1 %s temp.wav' % (self.sph2pipe, sph_dir[x] + 'audio/' + audio_fn))
 
            # Load .wav file
            _, y = wavfile.read('temp.wav')
            y_nonsil = []

            # Load segment intervals
            nonsil_whole_utterance = nonsilence_intervals[utt_id]    
            start_seg, end_seg = 0, 0
            for i_seg, seg_id in enumerate(sorted(nonsil_whole_utterance, key=lambda x:int(x))):
              nonsil_segment = nonsil_whole_utterance[seg_id]
              # XXX
              # if i_seg > 5:
              #   continue
              for start_end in nonsil_segment:          
                start = int(start_end[0] * self.fs)
                end = int(start_end[1] * self.fs) 
                seg = y[start:end] 
                y_nonsil.append(seg)
                end_seg += (start_end[1] - start_end[0])  
              
              segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_seg, end_seg))
              start_seg = end_seg

            y_nonsil = np.concatenate(y_nonsil, axis=0)

            # Segment the audio, save it as a .wav file
            wav_scp_f.write(utt_id + ' ' + os.path.join(out_dir, x, audio_fn.split('.')[0]+'.wav') + '\n')
            wavfile.write(os.path.join(out_dir, x, audio_fn.split('.')[0]+'.wav'), self.fs, y_nonsil) 

  def merge_short_utterances(self, out_tag='_merged', min_dur_sec=3, min_text_char=6):
    # XXX
    for x in ['train']:
      if not os.path.isdir(os.path.join('data', x+out_tag)): 
        os.mkdir(os.path.join('data', x+out_tag))
      
      with open(os.path.join('data', x+'_nonsil', 'segments'), 'r') as in_segment_f,\
           open(os.path.join('data', x+'_nonsil', 'text'), 'r') as in_text_f,\
           open(os.path.join('data', x+out_tag, 'segments'), 'w') as out_segment_f,\
           open(os.path.join('data', x+out_tag, 'text'), 'w') as out_text_f,\
           open(os.path.join('data', x+out_tag, 'utt2spk'), 'w') as utt2spk_f:
        out_segment_f.truncate()
        out_text_f.truncate()

        i_seg = 0 
        dur = 0.
        sent_len = 0
        start_seg = 0.
        sent_seg = ''
        lines_in_segment = in_segment_f.readlines()
        lines_in_text = in_text_f.readlines()
        prev_utt_id = ''
        for line_in_seg, line_in_text in zip(lines_in_segment, lines_in_text):
          parts = line_in_seg.strip().split()
          sent = ' '.join(line_in_text.strip().split()[1:])
          seg_id = parts[0]
          utt_id = parts[1]
          if prev_utt_id != utt_id:
            print(prev_utt_id, utt_id)
            i_seg = 0
            dur = 0.
            sent_len = 0
            start_seg = 0.
            sent_seg = ''
            prev_utt_id = deepcopy(utt_id)
          start = float(parts[2])
          end = float(parts[3])
          dur += (end - start)
          sent_len += len(sent.split())
          sent_seg = sent_seg + ' ' + sent

          if dur < min_dur_sec or sent_len < min_text_char:
            continue
          else:             
            # print(utt_id, sent_seg, dur, sent_len)
            out_segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_seg, start_seg + dur))
            out_text_f.write('%s_%04d %s\n' % (utt_id, i_seg, sent_seg)) 
            utt2spk_f.write('%s_%04d %s\n' % (utt_id, i_seg, utt_id))
            i_seg += 1
            start_seg += dur
            dur = 0.
            sent_len = 0
            sent_seg = ''

if __name__ == '__main__':
  data_root = '/home/lwang114/data/babel/IARPA_BABEL_BP_101/'
  # '/Users/liming/research/data/IARPA_BABEL_BP_101/'
  # '/home/lwang114/data/babel/IARPA_BABEL_BP_101/'
  exp_root = 'exp/apr1_BP1_101_conversational/'
  sph2pipe = '/home/lwang114/kaldi/tools/sph2pipe_v2.5/sph2pipe'
  # 'sph2pipe_v2.5/sph2pipe'
  # '/home/lwang114/kaldi/tools/sph2pipe_v2.5/sph2pipe'
  configs = {'audio_type': 'conversational', 'is_segment': True, 'vad': True}
  kaldi_prep = BabelKaldiPreparer(data_root, exp_root, sph2pipe, configs)
  # kaldi_prep.prepare_tts()
  kaldi_prep.remove_silence() 
  kaldi_prep.merge_short_utterances()
