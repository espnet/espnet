import os
import numpy as np
import librosa

UNK = ['(())']
NONWORD = '~'
EPS = 1e-3
LOGEPS = -60
DEBUG = True

def VAD(y, fs, thres=EPS, coeff=1.0):
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

  # print(start_sec_nonsils[:10], end_sec_nonsils[:10])
  # print(start_nonsils[:10], end_nonsils[:10])
  return start_sec_nonsils, end_sec_nonsils

def VAD2(y, fs, thres=EPS):
  L = y.shape[0]
  dur = float(y.shape[0] / fs)
  n_fft = int(25 * fs / 1000)
  hop_length = int(10 * fs / 1000)
  # win_len = 2
  
  sgram = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=40, n_fft=n_fft, hop_length=hop_length)
  # print(sgram.shape)
  energies = np.sum(sgram, axis=0)
  mask = (energies > thres).astype(float)
  mask_diff = np.diff(mask)
  start_nonsils = np.where(mask_diff > 0)[0] 
  end_nonsils = np.where(mask_diff < 0)[0]
  start_sec_nonsils = [float(start * hop_length) / fs for start in start_nonsils]
  end_sec_nonsils = [float(end * hop_length) / fs for end in end_nonsils]
  if len(end_sec_nonsils) < len(start_sec_nonsils):
    end_sec_nonsils.append(dur)

  # print(start_sec_nonsils[:10], end_sec_nonsils[:10])
  # print(start_nonsils[:10], end_nonsils[:10])
  return start_sec_nonsils, end_sec_nonsils

class BabelKaldiPreparer:
  def __init__(self, data_root, exp_root, sph2pipe, configs):
    self.audio_type = configs.get('audio_type', 'scripted')
    self.is_segment = configs.get('is_segment', False)
    self.vad = configs.get('vad', True)
    self.verbose = configs.get('verbose', 0)
    self.fs = 8000
    self.data_root = data_root
    self.exp_root = exp_root
    self.sph2pipe = sph2pipe

    if self.audio_type == 'conversational':
      # XXX Use sub-train
      self.transcripts = {'train': os.listdir(data_root+'conversational/sub-train/transcript_roman/'),
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
           open(os.path.join('data', x, 'segments'), 'w') as segment_f:  
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        segment_f.truncate()
 
        i = 0
        for transcript_fn, audio_fn in zip(sorted(self.transcripts[x], key=lambda x:x.split('.')[0]), sorted(self.audios[x], key=lambda x:x.split('.')[0])):
          # XXX
          # if i > 1:
          #   continue
          # i += 1

          # Load audio
          os.system('/home/lwang114/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s temp.wav' % (sph_dir[x] + 'audio/' + audio_fn))
          y, _ = librosa.load('temp.wav', sr=self.fs)  
          utt_id = transcript_fn.split('.')[0]
          sent = []
          if self.is_segment:
            print(x, utt_id)
            with open(sph_dir[x] + 'transcript_roman/' + transcript_fn, 'r') as transcript_f:
              lines = transcript_f.readlines()
              i_seg = 0
              for i_seg, (start, segment, end) in enumerate(zip(lines[::2], lines[1::2], lines[2::2])):
                # XXX
                # if i_seg > 1:
                #   continue
                # i_seg += 1

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
                  start_sec_nonsils, end_sec_nonsils = VAD2(np.append(np.zeros((start,)), y[start:end]), self.fs)
                  if len(start_sec_nonsils) == 0:
                    print('Audio too quiet')
                    continue
                  # for i_subseg, (start_sec_nonsil, end_sec_nonsil) in enumerate(zip(start_sec_nonsils, end_sec_nonsils)): 
                  segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_sec_nonsils[0], end_sec_nonsils[-1])) 
                else:
                  segment_f.write('%s_%04d %s %.2f %.2f\n' % (utt_id, i_seg, utt_id, start_sec, end_sec)) 

                text_f.write('%s_%04d %s\n' % (utt_id, i_seg, ' '.join(words)))            
                utt2spk_f.write('%s_%04d %s\n' % (utt_id, i_seg, utt_id)) # XXX dummy speaker id
            if len(sent) == 0:
              continue
            wav_scp_f.write(utt_id + ' ' + self.sph2pipe + ' -f wav -p -c 1 ' + \
                    os.path.join(sph_dir[x], 'audio/', utt_id + '.sph') + ' |\n')
          else:
            # TODO: Remove silence interval
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
            utt2spk_f.write(utt_id + ' ' + '001\n') # XXX dummy speaker id
      if not self.is_segment:
        os.remove(os.path.join('data', x, 'segments'))
  
  # TODO
  def remove_silence(self):
    feat_dir = 'fbank/'
    if not os.path.isdir('fbank_no_silence/'):
      os.mkdir('fbank_no_silence')

    ark_files = {
      'train': os.listdir(feat_dir+'*_train.ark'),
      'dev': os.listdir(feat_dir+'*_dev.ark'),
      'test': os.listdir(feat_dir+'*_test.ark')
      }
        

if __name__ == '__main__':
  data_root = '/home/lwang114/data/babel/IARPA_BABEL_BP_101/'
  exp_root = 'exp/apr1_BP1_101_conversational/'
  sph2pipe = '/home/lwang114/kaldi/tools/sph2pipe_v2.5/sph2pipe'
  configs = {'audio_type': 'conversational', 'is_segment': True, 'vad': False}
  kaldi_prep = BabelKaldiPreparer(data_root, exp_root, sph2pipe, configs)
  kaldi_prep.prepare_tts()
