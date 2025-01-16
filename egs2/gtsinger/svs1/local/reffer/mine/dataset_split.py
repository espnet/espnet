import argparse
import os
import shutil
import textgrid
import xmltodict
import json
import math
from pypinyin import Style, pinyin
from pypinyin.style._utils import get_finals, get_initials


from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader

UTT_PREFIX = "GTSINGER"
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
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial + "_" + final)
        else:
            phones.append(final)
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
        ##
        label_info.append(f"{start_time} {end_time} {label}")
        pho_info.append(label)
    
    return " ".join(label_info), " ".join(pho_info)


#处理musicxml文件获取score信息并在textfile记录歌词
def process_score_info(tempo, notes, textfile, wordtextfile):
    #tempo: 音频的节奏, 如143
    #notes: 这个列表包含了音符信息，每个音符有多个属性，如开始时间（st）、结束时间（et）、歌词（lyric）等,[<espnet2.fileio.score_scp.NOTE object at 0x7f25077913d0>,...]
    score_notes = []
    i, length = 0, len(notes)
    pre_phoneme = ' '
    while i < length:
        # Divide songs by 'P' (pause) or 'B' (breath) or GlottalStop
        #note[i] = <espnet2.fileio.score_scp.NOTE object at 0x7f07c26048b0>
        #note[i]包括lyric、midi、st、et: 你|58|3.780|4.146    P|0|5.365|5.731
        # fix errors in dataset
        # remove rest note
        # P:
        # -:表示歌词延续
        # SP是silence，AP是换气声（就吸一口气）  
        wordtextfile.write("{} ".format(notes[i].lyric)) 
        if notes[i].lyric == 'P':
            if notes[i].et - notes[i].st < 1:
                notes[i].lyric = 'AP'
            else:
                notes[i].lyric = 'SP'
        if notes[i].lyric == '—':
            notes[i].lyric = pre_phoneme
        if i < length:
            phn = pypinyin_g2p_phone_without_prosody(notes[i].lyric)[0]
            score_notes.append([notes[i].st, notes[i].et, notes[i].lyric, notes[i].midi, phn])
            phonemes = phn.split("_")
            pre_phoneme = phonemes[-1] 
            textfile.write(' '.join(f'{p}' for p in phonemes))
            textfile.write(' ')
        i += 1
    textfile.write("\n")
    wordtextfile.write("\n")
    return dict(tempo=tempo, item_list=["st", "et", "lyric", "midi","phn"], note=score_notes)


def cal_dur(self, duration, divisions, tempo):
    return float(duration) / float(divisions) * 60.0 / float(tempo)


#处理musicxml和textgrid
def tg_hier(tg_path, tempo, note_list): # tg_path = ***.textgrid
    print(tg_path)
    word_dict_list = []
    tg = textgrid.TextGrid.fromFile(tg_path+".TextGrid")
    word_intervals = tg[0] #word
    ph_intervals = tg[1]   #phn
    mix_intervals= tg[2]
    falsetto_intervals= tg[3]
    breathe_intervals= tg[4]
    pharyngeal_intervals= tg[5]
    vibrato_intervals= tg[6]
    glissando_intervals= tg[7]

    # get word information
    for word in word_intervals:
        mark = word.mark.strip()
        min=word.minTime
        max=word.maxTime
        word_dict = {}
        word_dict["word"] = mark
        word_dict["start_time"] = round(float(min),3)
        word_dict["end_time"] = round(float(max),3)
        word_dict["note"]=[] 
        word_dict['note_dur']=[]
        word_dict['note_start']=[]
        word_dict['note_end']=[]
        word_dict['ph']=[]
        word_dict['ph_start']=[]
        word_dict['ph_end']=[]    
        word_dict['mix']=[]
        word_dict['falsetto']=[]
        word_dict['breathe']=[]
        word_dict['pharyngeal']=[]
        word_dict['vibrato']=[]
        word_dict['glissando']=[] 
        word_dict['tech']=''
        word_dict_list.append(word_dict)

    mix_list=[]
    for t in mix_intervals:
        mix_list.append(t.mark)

    falsetto_list=[]
    for t in falsetto_intervals:
        falsetto_list.append(t.mark)

    breathe_list=[]
    for t in breathe_intervals:
        breathe_list.append(t.mark)

    pharyngeal_list=[]
    for t in pharyngeal_intervals:
        pharyngeal_list.append(t.mark)

    vibrato_list=[]
    for t in vibrato_intervals:
        vibrato_list.append(t.mark)

    glissando_list=[]
    for t in glissando_intervals:
        glissando_list.append(t.mark)    
    
    # do phoneme alignment
    #word_intervals = tg[0]
    #ph_intervals = tg[1]
    #word_dict_list=[{'word': '你', 'start_time': 0.0,...},{...},...]
    idx = 0
    for n,ph in enumerate(ph_intervals):# n=0   ph=Interval(0.0, 0.125, n)
        if idx >= len(word_dict_list):
            print(f'{ph}{idx}, word_dict_list length is {len(word_dict_list)}')
            for element in word_dict_list:
                print(f'{element["word"]},{element["ph"]}')
            assert False
        # ph = ph.strip()
        mark=ph.mark.strip()
        min=round(float(ph.minTime),3)
        max=round(float(ph.maxTime),3)
        
        if min>=word_dict_list[idx]['start_time'] and max<=word_dict_list[idx]['end_time']:
            word_dict_list[idx]['ph'].append(mark)
            word_dict_list[idx]['ph_start'].append(min)
            word_dict_list[idx]['ph_end'].append(max)
            word_dict_list[idx]['mix'].append(mix_list[n])
            word_dict_list[idx]['falsetto'].append(falsetto_list[n])
            word_dict_list[idx]['breathe'].append(breathe_list[n])
            word_dict_list[idx]['pharyngeal'].append(pharyngeal_list[n])
            word_dict_list[idx]['vibrato'].append(vibrato_list[n])
            word_dict_list[idx]['glissando'].append(glissando_list[n])
        else:
            idx = idx + 1
            if min>=word_dict_list[idx]['start_time'] and max<=word_dict_list[idx]['end_time']:
                word_dict_list[idx]['ph'].append(mark)
                word_dict_list[idx]['ph_start'].append(min)
                word_dict_list[idx]['ph_end'].append(max) 
                word_dict_list[idx]['mix'].append(mix_list[n])
                word_dict_list[idx]['falsetto'].append(falsetto_list[n])
                word_dict_list[idx]['breathe'].append(breathe_list[n])
                word_dict_list[idx]['pharyngeal'].append(pharyngeal_list[n])
                word_dict_list[idx]['vibrato'].append(vibrato_list[n])
                word_dict_list[idx]['glissando'].append(glissando_list[n])
        if word_dict_list[idx]['mix']=='1':
            word_dict_list[idx]['tech']+=' 1'
        if word_dict_list[idx]['falsetto']=='1':
            word_dict_list[idx]['tech']+=' 2'
        if word_dict_list[idx]['breathe']=='1':
            word_dict_list[idx]['tech']+=' 3'
        if word_dict_list[idx]['pharyngeal']=='1':
            word_dict_list[idx]['tech']+=' 4'
        if word_dict_list[idx]['vibrato']=='1':
            word_dict_list[idx]['tech']+=' 5'
        if word_dict_list[idx]['glissando']=='1':
            word_dict_list[idx]['tech']+=' 6'
          

    assert idx == len(word_dict_list)-1, f"ph not align, {idx} != {len(word_dict_list)}"

    # get tempo and note information
    xml_fn=tg_path + '.musicxml'

    with open(xml_fn, "r") as f:
        xml_content = f.read()
    #print(xml_content)
    data_dict = xmltodict.parse(xml_content)
    divisions = []
    #data_dict -> score-partwise
    #score-partwise -> work,identification,part-list,rument,part
    #part -> @id,measure(列表)
    print(len(note_list))
    print(len(word_dict_list))
    
    print("-------------------")
    for msg in data_dict['score-partwise']['part']['measure']:
        #msg = {
        # @number :: 3
        # attributes :: {'divisions': '2'}
        # note :: [{'pitch': {'step': 'C', 'alter': '1', 'octave': '4'}, 'duration': '1', 'tie': {'@type': 'stop'}, 'voice': '1', 'type': 'eighth', 'notations': {'tied': {'@type': 'stop'}}}, {'pitch': {'step': 'C', 'alter': '0', 'octave': '4'}, 'duration': '2', 'voice': '1', 'type': 'quarter', 'lyric': {'syllabic': 'single', 'text': '深'}}, {'pitch': {'step': 'G', 'alter': '0', 'octave': '3'}, 'duration': '1', 'voice': '1', 'type': 'eighth', 'lyric': {'syllabic': 'single', 'text': '的'}}, {'pitch': {'step': 'F', 'alter': '0', 'octave': '3'}, 'duration': '2', 'voice': '1', 'type': 'quarter', 'lyric': {'syllabic': 'single', 'text': '脑'}}, {'pitch': {'step': 'D', 'alter': '1', 'octave': '3'}, 'duration': '2', 'tie': {'@type': 'start'}, 'voice': '1', 'type': 'quarter', 'notations': {'tied': {'@type': 'start'}}, 'lyric': {'syllabic': 'single', 'text': '海'}}]
        print(":",len(msg['note']))
        for k,v in msg.items():
            print(k,"::",v)
    '''
    # 转换 XML 为 JSON
    try:
        data_dict = xmltodict.parse(xml_content)
        json_content = json.dumps(data_dict, indent=2)
        print("Converted JSON:", json_content)
    except Exception as e:
        print(f"Error converting XML to JSON: {e}")
    
    #print(json_content)

    #with open(xml_fn, "r") as f:
    #    score_list = json.load(f)
    #print(score_list)
    divisions = int(json_content[0]["divisions"])
    tempo = float(json_content[1]["sound"]["@tempo"])
    note_list = json_content[2:]'''


    idx=0
    for note in note_list:
        if note['lyric']== '' and note['slur']==[]: #slur 指的是一种连音符号
            word_dict_list[idx]['note'].append(0)
            word_dict_list[idx]['note_dur'].append(cal_dur(note["duration"], divisions, tempo))
        elif note['lyric']==word_dict_list[idx]['word'] or (note['lyric']== '' and note['slur']!=[]):
            word_dict_list[idx]['note'].append(note['pitch'])
            word_dict_list[idx]['note_dur'].append(cal_dur(note["duration"], divisions, tempo))            
        if note['slur']==[] or note['slur']==['stop']:
            idx+=1

    for word in word_dict_list:
        time=word['start_time']
        for note in word['note']:
            dur=word['note_dur']/(word['end_time']-word['start_time'])
            word['note_start']=time
            word['note_end']=time+dur
            time=time+dur

    with open(tg_path.replace(f".TextGrid", ".json"), 'w', encoding="utf8") as f:
        json.dump(word_dict_list, f, indent=4, ensure_ascii=False)


#src_data = /data3/tyx/dataset/GTSinger/Chinese
def process_subset(src_data, subset, check_func, fs, wav_dump, score_dump):
    singerfolder = os.listdir(src_data) #女中音（Alto）、男高音（Tenor）
    makedir(subset)
    wavscp = open(os.path.join(subset, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(subset, "utt2spk"), "w", encoding="utf-8")
    label_scp = open(os.path.join(subset, "label"), "w", encoding="utf-8")
    musicxmlscp = open(os.path.join(subset, "score.scp"), "w", encoding="utf-8")

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
                    #key = Alto1_Breathy_不再见_0001
                    key = "{}{}_{}_{}_{}".format(sifolder.split("-")[1],sifolder.split("-")[2],skfolder,sofolder,str(i).zfill(4))
                    
                    #Breathy_Group：帮助模型学习带有情感特征的演唱方式
                    #Control_Group：提供标准化的对照数据，用于基础模型训练
                    #Paired_Speech_Group：为语音到歌声（speech-to-singing）任务或歌声分析提供对应的语音数据

                    #path = /data3/tyx/dataset/GTSinger/Chinese/ZH-Alto-1/Pharyngeal/老男孩/Control_Group/0000
                    path = os.path.join(src_data,sifolder,skfolder,sofolder,"Control_Group",str(i).zfill(4))
                    if not os.path.exists(path + ".wav"):
                        continue
                    utt_id = "{}_{}".format(UTT_PREFIX,key) #GTSINGER_Alto1_Breathy_不再见_0001
                    '''
                    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}_bits16.wav".format(
                        path,
                        fs,
                        os.path.join(wav_dump, key),
                    )
                    os.system(cmd)
                    '''
                    wavscp.write("{} {}\n".format(utt_id, os.path.join(wav_dump, "{}_bits16.wav".format(key)))) 
                    utt2spk.write("{} {}\n".format(utt_id, sifolder.split("-")[1] + sifolder.split("-")[2])) #utt_id Alto1
                    musicxmlscp.write("{} {}\n".format(utt_id, path + ".musicxml"))
                    
                    #abel_info, pho_info = process_pho_info(path + ".TextGrid") #label_info是列表
                    #label_scp.write("{} {}\n".format(utt_id, label_info))

    #step2 : 处理 score.scp, text ：musicxml->json
    reader = XMLReader(os.path.join(subset, "score.scp"))
    scorescp = open(os.path.join(subset, "score.scp"), "r", encoding="utf-8")
    score_writer = SingingScoreWriter(score_dump, os.path.join(subset, "score.scp.tmp"))
    text = open(os.path.join(subset, "text"), "w", encoding="utf-8")
    word_text = open(os.path.join(subset, "wordtext"), "w", encoding="utf-8")
    #xml_line = GTSINGER_Tenor1_Glissando_我的歌声里_0000 /data3/tyx/.../Control_Group/0000.musicxml
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        tempo, temp_info = reader[xmlline[0]] #xmlline[0]=GTSINGER_Tenor1_Glissando_我的歌声里_0000
        
        keypath = os.path.splitext(xmlline[1])[0] #/data3/tyx/.../Control_Group/0000
        tg_hier(keypath, tempo, temp_info)
        
        text.write("{} ".format(xmlline[0]))
        word_text.write("{} ".format(xmlline[0]))
        score_info = process_score_info(tempo, temp_info, text, word_text)
        score_writer[xmlline[0]] = score_info #score_info是字典



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Oniku Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    parser.add_argument("--fs", type=int, help="frame rate (Hz)")
    parser.add_argument(
        "--wav_dump", type=str, default="wav_dump", help="wav dump directory"
    )
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav_dump):
        os.makedirs(args.wav_dump)

    process_subset(args.src_data, args.train, train_check, args.fs, args.wav_dump, args.score_dump)
    process_subset(args.src_data, args.dev, dev_check, args.fs, args.wav_dump, args.score_dump)
    process_subset(args.src_data, args.test, test_check, args.fs, args.wav_dump, args.score_dump)
