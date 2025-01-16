import argparse
import os
import ast
import shutil
import textgrid
import xmltodict
import json
import math
from pypinyin import Style, pinyin
from pypinyin.style._utils import get_finals, get_initials

from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader


UTT_PREFIX = "GTSINGER_CHINESE"
DEV_LIST = [
    "不再见",
    "曹操",
    "爱情转移",
    "大鱼",
    "安河桥",
]
TEST_LIST = [
    "匆匆那年",
    "可惜没如果",
    "菊花台",
    "默",
    "画心",
]


yue_song_list = []   #歌曲列表，在这些歌曲中 "乐" 发音为 "yue"，标记为 "ve"
unique_label_dict = {}  #一些特殊的标注需要重新处理


def pre_unique_data(yue_songs_file, unique_label_file):
    #yue_songs_list = ["Chinese/ZH-Tenor-1/Vibrato/我真的受伤了/Control_Group/0008",...]
    with open(yue_songs_file, 'r', encoding='utf-8') as file:
        for line in file:
            yue_song_list.append(line.strip())

    #song_label_dict = {"Chinese/ZH-Tenor-1/Vibrato/我真的受伤了/Control_Group/0008":[pho_info, label_info]}
    with open(unique_label_file, 'r', encoding='utf-8') as file:
        index = 0
        key = ""
        for line in file:
            if len(line) < 2:
                break
            if index % 3 == 0:
                key = line.strip()
                unique_label_dict[key] = []
            else:
                unique_label_dict[key].append(ast.literal_eval(line))  
            index += 1

    
def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


def pypinyin_g2p_phone_without_prosody(text):
    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        #print("::",phone," : ",initial," : ",final)
        #y_in in    y_i i  y_uan van  y_ue ve 
        #y_ong iong  y_ou iou  y_an ian 
        #w_o uo
        #w_ei uei  h_ui h_uei
        if len(initial) != 0:
            if initial == "y":
                if final in ["in", "i", "uan", "ue", "ing", "u"]:
                    if final == "uan":
                        final = "van"
                    if final == "ue":
                        final = "ve"
                    if final == "u":
                        final == "v"
                else:
                    final = "i" + final
                phones.append(final)
                continue
            elif initial == "w":
                if final == "o":
                    final = "uo"
                if final == "an":
                    final = "uan"
                phones.append(final)
                continue
            elif initial == "c":
                if final == "un":
                    final = "uen"
            elif initial in ["x", "y", "j", "q", "l"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
                elif final == "ue":
                    final = "ve"
                elif final == "iu":
                    final = "iou"
            elif final == "ui":
                final = "uei"
            phones.append(initial + "_" + final)
        else:
            phones.append(final)
    if text == "喔":
        phones = ["w_o"]
    return phones


#处理textgrid文件获取label信息
def process_pho_info(filepath):
    tg = textgrid.TextGrid.fromFile(filepath)
    #获取 'phone' 标注层 [label]
    phone_tier = None
    for tier in tg.tiers:
        if tier.name == 'phone':
            phone_tier = tier
            break
    if phone_tier is None:
        raise ValueError("No 'phone' tier found in the TextGrid file.")
    label_info = []
    pho_info = []
    for interval in phone_tier:
        start_time = interval.minTime    #使用 minTime 表示开始时间
        end_time = interval.maxTime      #使用 maxTime 表示结束时间
        label = interval.mark.strip()    #获取标签
        if '<' in label or '>' in label: #处理类似 <SP> <AP>，将 <SP> 变为 SP
            label = label[1:-1]
        label_info.append(f"{start_time} {end_time} {label}")
        pho_info.append(label)

    return label_info, pho_info


#处理notes获取score信息和歌词
def process_score_info(notes, label_pho_info, utt_id, label = None):
    #tempo: 音频的节奏, 如143
    #notes: 这个列表包含了音符信息，每个音符有多个属性，如开始时间（st）、结束时间（et）、歌词（lyric）等,[<espnet2.fileio.score_scp.NOTE object at 0x7f25077913d0>,...]
    word_text_list = []
    score_notes = []
    phnes = []
    labelind = 0
    for i in range(len(notes)):
        # Divide songs by 'P' (pause) or 'B' (breath) or GlottalStop
        # fix errors in dataset
        # remove rest note
        #note[i] = <espnet2.fileio.score_scp.NOTE object at 0x7f07c26048b0>
        #note[i]包括lyric、midi、st、et: 你|58|3.780|4.146    P|0|5.365|5.731
        # —:表示歌词延续
        # SP是silence，AP是换气声（吸一口气）  
        word_text_list.append(notes[i].lyric)
        if notes[i].lyric == "—":
            score_notes[-1][1] = notes[i].et
        if notes[i].lyric == "P":
            notes[i].lyric = "AP"
        if notes[i].lyric != "—":
            phonemes = pypinyin_g2p_phone_without_prosody(notes[i].lyric)[0].split("_")
            if notes[i].lyric == "乐" and utt_id in yue_song_list:
                phonemes = ["ve"]
            for j in range(len(phonemes)):
                if labelind >= len(label_pho_info): #error
                    print("error.................i=",i," len(notes)=",len(notes))
                    print(notes[i].lyric, " phonemes=",phonemes, "j=", j)
                    print("labelind=",labelind)
                    print(utt_id)
                    print(label_pho_info)
                    print(label)
                    exit(1)
                phonemes[j] = label_pho_info[labelind]
                labelind += 1
            #print(notes[i].lyric, "   ", phonemes)
            score_notes.append([notes[i].st, notes[i].et, notes[i].lyric, notes[i].midi, "_".join(phonemes)]) 
            phnes.extend(phonemes)
    
    return score_notes, word_text_list, phnes


#处理textgrid文件获取label
#处理musicxml文件获取score
def process_json_to_pho_score(basepath, tempo, notes):
    #basepath=/data3/tyx/dataset/GTSinger/Chinese/ZH-Alto-1/Pharyngeal/老男孩/Control_Group/0000 
    parts = basepath.split('/')
    utt_id = '/'.join(parts[-6:]) #Chinese/ZH-Alto-1/Pharyngeal/老男孩/Control_Group/0000
    if utt_id in unique_label_dict.keys():
        pho_info = unique_label_dict[utt_id][0]
        label_info = unique_label_dict[utt_id][1]
    else :
        label_info, pho_info = process_pho_info(basepath + '.TextGrid')

    score_notes, word_text_list, phnes = process_score_info(notes, pho_info, utt_id, label_info)

    if len(pho_info) != len(phnes): #error
        print("erro....mismatch(label and score), ")
        print(utt_id)
        print(pho_info)
        print(label_info)
        print("------------phnes=")
        print(phnes)
        print("-----------word_text_lists=")
        print(word_text_list)
        print("-----------label_info=")
        print(label_info)
        print("-----------score_notes=")
        print(score_notes)
        exit(1)
    else: #check score and label
        sign = True
        f = False
        for i in range(len(pho_info)):
            assert pho_info[i] == phnes[i]
            if pho_info[i] != phnes[i]:
                f = True
                if sign:
                    print(utt_id)
                    print(pho_info)
                    print(label_info)
                    print(word_text_list)
                    sign = False
                print("mismatch[{}]: {} != {}".format(i,pho_info[i],phnes[i]))
        if f == True:
            print("-----------label_info=")
            print(label_info)
            print("-----------score_notes=")
            print(score_notes)
            exit(1)
    
    return " ".join(label_info), \
            " ".join(pho_info), \
            " ".join(word_text_list), \
            dict(tempo=tempo, item_list=["st", "et", "lyric", "midi", "phn"], note=score_notes) 


#src_data = /data3/tyx/dataset/GTSinger/Chinese
def process_subset(src_data, subset, check_func, fs, wav_dump, score_dump):
    singerfolder = os.listdir(src_data) #女中音（Alto）、男高音（Tenor）
    makedir(subset)
    wavscp = open(os.path.join(subset, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(subset, "utt2spk"), "w", encoding="utf-8")
    label_scp = open(os.path.join(subset, "label"), "w", encoding="utf-8")
    musicxml = open(os.path.join(subset, "score.scp"), "w", encoding="utf-8")

    #step1 : 处理 label, utt2spk, wav.scp, score.scp(utt_id, musicxml)
    #sifolder = 'ZH-Alto-1','ZH-Tenor-1'
    for sifolder in singerfolder:
        #Breathy：气声  Glissando：滑音 
        #Mixed_Voice_and_Falsetto：混声与假声
        #Pharyngeal：咽音 Vibrato：颤音
        skillfolder = os.listdir(os.path.join(src_data,sifolder))

        #skfolder = 'Breathy','Glissando',...    
        for skfolder in skillfolder:
            songfolder = os.listdir(os.path.join(src_data,sifolder,skfolder))

            #sofolder = '不再见','成都',...
            for sofolder in songfolder:
                if not check_func(sofolder):
                    continue
                for i in range(12): #0,1,2,3,...,11 
                    #Breathy_Group：帮助模型学习带有情感特征的演唱方式
                    #Control_Group：提供标准化的对照数据，用于基础模型训练
                    #Paired_Speech_Group：为语音到歌声（speech-to-singing）任务或歌声分析提供对应的语音数据
                    for group in ["Breathy", "Control"]:
                        #key = Alto1_Breathy_不再见_BreathyGroup_0001
                        key = "{}{}_{}_{}_{}_{}".format(sifolder.split("-")[1],sifolder.split("-")[2],skfolder,sofolder,group+"Group",str(i).zfill(4))
                    
                        #path = /data3/tyx/dataset/GTSinger/Chinese/ZH-Alto-1/Pharyngeal/老男孩/Control_Group/0000
                        path = os.path.join(src_data,sifolder,skfolder,sofolder,group+"_Group",str(i).zfill(4))
                        if not os.path.exists(path + ".wav"):
                            continue
                        utt_id = "{}_{}".format(UTT_PREFIX,key) #GTSINGER_Alto1_Breathy_不再见_BreathyGroup_0001
                        
                        cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}.wav".format(
                            path,
                            fs,
                            os.path.join(wav_dump, utt_id)
                        )
                        os.system(cmd)
                        
                        wavscp.write("{} {}\n".format(utt_id, os.path.join(wav_dump, "{}.wav".format(utt_id)))) 
                        utt2spk.write("{} {}\n".format(utt_id, sifolder.split("-")[1] + sifolder.split("-")[2])) #utt_id Alto1
                        musicxml.write("{} {}\n".format(utt_id, path + ".musicxml"))

    
    #step2 : 处理 score.scp, text : musicxml->json
    reader = XMLReader(os.path.join(subset, "score.scp"))
    scorescp = open(os.path.join(subset, "score.scp"), "r", encoding="utf-8")
    score_writer = SingingScoreWriter(score_dump, os.path.join(subset, "score.scp.tmp"))
    text = open(os.path.join(subset, "text"), "w", encoding="utf-8")
    word_text = open(os.path.join(subset, "wordtext"), "w", encoding="utf-8")
    #xml_line = GTSINGER_Tenor1_Glissando_我的歌声里_BreathyGroup_0000 /data3/tyx/.../Breathy_Group/0000.musicxml
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        tempo, tempo_info = reader[xmlline[0]] #xmlline[0]=GTSINGER_Tenor1_Glissando_我的歌声里_BreathyGroup_0000        
        basepath = os.path.splitext(xmlline[1])[0] #/data3/tyx/.../Breathy_Group/0000
        label_info, text_info, word_text_info, score_info = process_json_to_pho_score(basepath, tempo, tempo_info)

        label_scp.write("{} {}\n".format(xmlline[0], label_info))
        text.write("{} {}\n".format(xmlline[0], text_info))
        word_text.write("{} {}\n".format(xmlline[0], word_text_info))
        score_writer[xmlline[0]] = score_info #score_info是字典


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Oniku Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    parser.add_argument("--fs", type=int, help="frame rate (Hz)")
    parser.add_argument("--wav_dump", type=str, default="wav_dump", help="wav dump directory")
    parser.add_argument("--score_dump", type=str, default="score_dump", help="score dump directory")
    parser.add_argument("--yue_songs_file", type=str, default="./local/yue_songs.txt", help="song list file that \'月\' is pronounced \'yue\'")
    parser.add_argument("--unique_label_file", type=str, default="./local/unique_label.txt", help="unique song-label dict file")
    
    args = parser.parse_args()

    if not os.path.exists(args.wav_dump):
        os.makedirs(args.wav_dump)

    pre_unique_data(args.yue_songs_file, args.unique_label_file)

    process_subset(args.src_data, args.train, train_check, args.fs, args.wav_dump, args.score_dump)
    process_subset(args.src_data, args.dev, dev_check, args.fs, args.wav_dump, args.score_dump)
    process_subset(args.src_data, args.test, test_check, args.fs, args.wav_dump, args.score_dump)
