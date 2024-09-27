import os
import tqdm
import sys
import glob
import argparse
os.chdir(sys.path[0])

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav_source_dir', type=str, default='/nfs-02/yuyue/visualtts/dataset/lrs2/wav_16k')
    parser.add_argument('--video_source_dir', type=str, default='/nfs-02/yuyue/visualtts/dataset/lrs2/av_feature')
    parser.add_argument('--data_split_dir', type=str, default='/nfs-02/yuyue/visualtts/dataset/lrs2/data_split')

    args = parser.parse_args()

    return args

# use all data, output 4 docs: text, utt2spk, wav.scp, video.scp
if __name__ == "__main__":
    # 用上所有的data，val和test不变，train增加
    meta_data_dir = '../data'
    os.makedirs(meta_data_dir, exist_ok=True)

    args = parse_arguments()
    
    all_data_list = []
    for data_split in ["val", "test"]:
        if not os.path.exists(os.path.join(args.data_split_dir, data_split) + '.txt'):
            print(f"No data_split named {data_split}")

        info_list = []
        data_split_folder = os.path.join(meta_data_dir, data_split)
        os.makedirs(data_split_folder, exist_ok=True)

        with open(os.path.join(args.data_split_dir, data_split + '.txt'), 'r') as file:
            utt_list = file.readlines()
            utt_list = sorted(utt_list)
        for utt in tqdm.tqdm(utt_list):
            utt = utt.strip()
            count = len(utt_list)
            if os.path.exists(os.path.join(args.wav_source_dir, utt + '.wav')) and os.path.exists(os.path.join(args.wav_source_dir, utt + '.lab')):
                try:
                    with open(os.path.join(args.wav_source_dir, utt + '.lab'), 'r') as file:
                        text = file.readline().strip()
                    spk = utt.split('_')[0]
                    wav_path = os.path.join(args.wav_source_dir, utt + '.wav')
                    video_feature_path = os.path.join(args.video_source_dir, utt + '.mp4')
                    info_list.append((utt, text, spk, wav_path, video_feature_path))
                    all_data_list.append(utt)
                except:
                    continue
        #_name
        text = open(os.path.join(data_split_folder, 'text'), 'w')
        utt2spk = open(os.path.join(data_split_folder, 'utt2spk'), 'w')
        wav_scp = open(os.path.join(data_split_folder, 'wav.scp'), 'w')
        video_scp = open(os.path.join(data_split_folder, 'video.scp'), 'w')
        for info in info_list:
            text.write(f"{info[0]} {info[1]}\n")
            utt2spk.write(f"{info[0]} {info[2]}\n")
            wav_scp.write(f"{info[0]} {info[3]}\n")
            video_scp.write(f"{info[0]} {info[4]}\n")
        text.close()
        utt2spk.close()
        wav_scp.close()
        video_scp.close()
    
    data_split_folder = os.path.join(meta_data_dir, "train")
    os.makedirs(data_split_folder, exist_ok=True)
    wav_path_list = sorted(glob.glob(os.path.join(args.wav_source_dir, '*.wav')))
    info_list = []
    for wav_path in tqdm.tqdm(wav_path_list):
        utt = os.path.basename(wav_path).replace('.wav', '')
        if utt not in all_data_list:  
            if os.path.exists(os.path.join(args.wav_source_dir, utt + '.wav')) and os.path.exists(os.path.join(args.wav_source_dir, utt + '.lab')):
                try:
                    with open(os.path.join(args.wav_source_dir, utt + '.lab'), 'r') as file:
                        text = file.readline().strip()
                    spk = utt.split('_')[0]
                    wav_path = os.path.join(args.wav_source_dir, utt + '.wav')
                    video_feature_path = os.path.join(args.video_source_dir, utt + '.mp4')
                    info_list.append((utt, text, spk, wav_path, video_feature_path))
                    all_data_list.append(utt)
                except:
                    continue
    text = open(os.path.join(data_split_folder, 'text'), 'w')
    utt2spk = open(os.path.join(data_split_folder, 'utt2spk'), 'w')
    wav_scp = open(os.path.join(data_split_folder, 'wav.scp'), 'w')
    video_scp = open(os.path.join(data_split_folder, 'video.scp'), 'w')
    for info in info_list:
        text.write(f"{info[0]} {info[1]}\n")
        utt2spk.write(f"{info[0]} {info[2]}\n")
        wav_scp.write(f"{info[0]} {info[3]}\n")
        video_scp.write(f"{info[0]} {info[4]}\n")
    text.close()
    utt2spk.close()
    wav_scp.close()
    video_scp.close()

    with open(os.path.join(meta_data_dir, 'all.txt'), 'w') as file:
        for utt in all_data_list:
            file.write(f"{utt}\n")


# if __name__ == "__main__":
#     meta_data_dir = '../data'  # save_dir
#     os.makedirs(meta_data_dir, exist_ok=True)

#     args = parse_arguments()
    
#     for data_split in ["train", "val", "test"]:
#         if not os.path.exists(os.path.join(args.data_split_dir, data_split) + '.txt'):
#             print(f"No data_split named {data_split}")

#         info_list = []
#         data_split_folder = os.path.join(meta_data_dir, data_split)
#         os.makedirs(data_split_folder, exist_ok=True)
#         with open(os.path.join(args.data_split_dir, data_split + '.txt'), 'r') as file:
#             utt_list = file.readlines()
#             utt_list = sorted(utt_list)
#         for utt in utt_list:
#             utt = utt.strip()
#             if os.path.exists(os.path.join(args.wav_source_dir, utt + '.wav')) and os.path.exists(os.path.join(args.wav_source_dir, utt + '.lab')):
#                 try:
#                     with open(os.path.join(args.wav_source_dir, utt + '.lab'), 'r') as file:
#                         text = file.readline().strip()
#                     spk = utt.split('_')[0]
#                     wav_path = os.path.join(args.wav_source_dir, utt + '.wav')
#                     video_feature_path = os.path.join(args.source_video_feature_dir, utt + '.npy')
#                     info_list.append((utt, text, spk, wav_path, video_feature_path))
#                 except:
#                     continue
#         #_name
#         text = open(os.path.join(data_split_folder, 'text'), 'w')
#         utt2spk = open(os.path.join(data_split_folder, 'utt2spk'), 'w')
#         wav_scp = open(os.path.join(data_split_folder, 'wav.scp'), 'w')
#         video_scp = open(os.path.join(data_split_folder, 'video.scp'), 'w')
#         for info in info_list:
#             text.write(f"{info[0]} {info[1]}\n")
#             utt2spk.write(f"{info[0]} {info[2]}\n")
#             wav_scp.write(f"{info[0]} {info[3]}\n")
#             video_scp.write(f"{info[0]} {info[4]}\n")
#         text.close()
#         utt2spk.close()
#         wav_scp.close()
#         video_scp.close()

        





    
