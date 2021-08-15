import os
import glob
import shutil
import tqdm

if __name__ == '__main__':

    num_sample = 1000
    dataset_path = "/home/hdd1/20210528_jp_youtube-pilot/single-speaker"
    segment_path = "/home/hdd1/20210528_jp_youtube-pilot/single-speaker/100G_ctcscore_0622/segments.txt"
    test_paths = glob.glob(os.path.join(dataset_path, 'txt', '*', '*.txt'))[:num_sample]
    test_stems = [os.path.splitext(os.path.basename(stem))[0] for stem in test_paths]

    os.makedirs("/home/hdd1/jtuberaw/wav16k", exist_ok=True)
    os.makedirs("/home/hdd1/jtuberaw/txt", exist_ok=True)

    for path in tqdm.tqdm(test_paths):
        dirname = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        os.makedirs("/home/hdd1/jtuberaw/wav16k/{}".format(dirname), exist_ok=True)
        os.makedirs("/home/hdd1/jtuberaw/txt/{}".format(dirname), exist_ok=True)
        shutil.copy(
            os.path.join(dataset_path, 'txt', dirname, '{}.txt'.format(stem)),
            os.path.join("/home/hdd1/jtuberaw/txt", dirname, '{}.txt'.format(stem))    
        )
        shutil.copy(
            os.path.join(dataset_path, 'wav16k', dirname, '{}.wav'.format(stem)),
            os.path.join("/home/hdd1/jtuberaw/wav16k", dirname, '{}.wav'.format(stem))    
        )
    shutil.copy(
        os.path.join(dataset_path, "100G_ctcscore_0622/segments.txt"),
        os.path.join("/home/hdd1/jtuberaw/ctcscore.txt")    
    )