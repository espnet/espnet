import random
from pathlib import Path
from typing import Union
import argparse 
import shutil

def read_text(text: Path):
    uttid2text = {}
    with text.open("r") as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid2text[line[0]] = ' '.join(line[1:])
    return uttid2text


# Convert discrete unit to CJK characters
# CJK Unified Ideographs (20,977 chars): \u4e00 - \u9fff
def unit2cjk(unit: Union[str, int]) -> str:
    return chr(int('4e00', 16) + int(unit))

def cjk2unit(char: str) -> str:
    return str(ord(char) - int('4e00', 16))

def prepare_textlm(root: Path, out_dir=Path("data")):
    print("textlm:", root / 'text')
    uttid2text = read_text(root / 'text')
    res = []

    for uttid, text in uttid2text.items():
        uttid = f'textlm_{uttid}'
        text = text.lower()
        
        res.append(
            f"{uttid} <generatetext> {text}"
        )

    # write res
    print("Creating textlm: ", out_dir / 'lm_text')
    with (out_dir / 'lm_text').open('a') as fp:
        for line in res:
            fp.write(f'{line}\n')

    return

def prepare_speechlm(root: Path, out_dir=Path("data")):
    res = []
    uttid2token = read_text(root / f'token')    
    for uttid in uttid2token:
        token = uttid2token[uttid].split()
        token = [unit2cjk(t) for t in token]
        uttid = f'unitlm_{uttid}'
        token = ''.join(token)
        
        res.append(
            f"{uttid} <generatespeech>{token}"
        )

    print("Creating speechlm: ", out_dir / 'lm_text')
    with (out_dir / 'lm_text').open('a') as fp:
        for line in res:
            fp.write(f'{line}\n')

    return

def prepare_asr(root: Path, out_dir=Path("data")):
    uttid2text = read_text(root / f'text')
    uttid2token = read_text(root / f'token')
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        token = [unit2cjk(t) for t in token]
        
        uttid = f'asr_{uttid}'
        text = text.lower()
        token = ''.join(token)
       
        res.append(
            f"{uttid} <startofspeech>{token}<generatetext> {text}"
        )

    # write res
    print("Creating asr: ", out_dir / 'lm_text')
    with (out_dir / 'lm_text').open('a') as fp:
        for line in res:
            fp.write(f'{line}\n')

    return


def prepare_tts(root: Path, dset="train", out_dir=Path("data")):
    uttid2text = read_text(root / f'text')
    uttid2token = read_text(root / f'token')   
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        token = [unit2cjk(t) for t in token]
        
        uttid = f'tts_{uttid}'
        text = text.lower()
        token = ''.join(token)

        res.append(
            f"{uttid} <startoftext> {text}<generatespeech>{token}"
        )

    print("Creating tts: ", out_dir / 'lm_text')
    with (out_dir / 'lm_text').open('a') as fp:
        for line in res:
            fp.write(f'{line}\n')
    
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, help=" ", default="dump")
    args = parser.parse_args()
    out_dir = Path(args.path)

    #for dset in ['train', 'valid', 'test']:
    #for dset in ['valid', 'test']:
    #cur_dir = out_dir / dset
    
    # create empty file
    with (out_dir / 'lm_text').open('w') as fp:
        print("Opened file:",out_dir)

    #prepare textlm
    prepare_textlm(out_dir/ "text/textlm", out_dir=out_dir)

    # process speechlm
    prepare_speechlm(out_dir / "speech/speechlm", out_dir=out_dir)

    # process asr
    prepare_asr(out_dir / "speech/asr", out_dir=out_dir)

    # process tts
    prepare_tts(out_dir / "speech/tts", out_dir=out_dir)

        

    
    # for dset in ['train', 'valid', 'test']:
    #     cur_dir = out_dir / dset       
    #     shutil.copyfile(cur_dir /f'text', cur_dir /f'text_old')
    #     shutil.move(cur_dir /f'lm_text', cur_dir /f'text')
    
    with (out_dir / 'nlsyms.txt').open('w') as fp:
        fp.write('<startofspeech>\n<generatetext>\n<startoftext>\n<generatespeech>\n')
