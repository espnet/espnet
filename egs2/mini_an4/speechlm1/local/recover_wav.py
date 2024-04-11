import kaldiio 
import sys
import soundfile

uttid = sys.argv[1]
idx = sys.argv[2]
path = sys.argv[3]

fs, wav = kaldiio.load_mat(idx)
print(fs, wav.shape, wav)
soundfile.write(f"{path}/{uttid}.wav", wav, fs)