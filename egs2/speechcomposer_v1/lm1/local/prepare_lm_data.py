import argparse
import random
import shutil
from pathlib import Path
from typing import Union

from espnet2.utils.types import str2bool


def read_text(text: Path):
    uttid2text = {}
    with text.open("r") as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid2text[line[0]] = " ".join(line[1:])
    return uttid2text


# Convert discrete unit to CJK characters
# CJK Unified Ideographs (20,977 chars): \u4e00 - \u9fff
def unit2cjk(unit: Union[str, int]) -> str:
    return chr(int("4e00", 16) + int(unit))


def cjk2unit(char: str) -> str:
    return str(ord(char) - int("4e00", 16))


def prepare_textlm(
    root: Path, out_dir=Path("data"), generate_text_token="<generatetext>"
):
    print("textlm:", root / "text")
    uttid2text = read_text(root / "text")
    res = []

    for uttid, text in uttid2text.items():
        uttid = f"textlm_{uttid}"
        text = text.lower()

        res.append(f"{uttid} {generate_text_token} {text}")

    # write res
    print("Creating textlm: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_speechlm(
    root: Path,
    out_dir=Path("data"),
    generate_speech_token="<generatespeech>",
    use_cjk=True,
):
    res = []
    uttid2token = read_text(root / f"token")
    for uttid in uttid2token:
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]
        uttid = f"unitlm_{uttid}"
        token = "".join(token)

        res.append(f"{uttid} {generate_speech_token} {token}")

    print("Creating speechlm: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_asr(
    root: Path,
    out_dir=Path("data"),
    start_speech_token="<startofspeech>",
    generate_text_token="<generatetext>",
    use_cjk=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token = read_text(root / f"token")
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]

        uttid = f"asr_{uttid}"
        text = text.lower()
        token = "".join(token)

        res.append(f"{uttid} {start_speech_token}{token}{generate_text_token} {text}")

    # write res
    print("Creating asr: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_tts(
    root: Path,
    out_dir=Path("data"),
    enrollment_dict=enrollment_dict,
    start_text_token="<startoftext>",
    generate_speech_token="<generatespeech>",
    enrollment_speech_token="<enrollspeech>",
    use_cjk=True,
    use_enroll_speech=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token = read_text(root / f"token")
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]

        if use_enroll_speech:
            spk_id = uttid.split('_')[0]
            enroll_token = enrollspeech_dict[spk_id].split()
            if use_cjk:
                enroll_token = [unit2cjk(t) for t in enroll_token]
                enroll_token = ''.join(enroll_token)
            
        uttid = f"tts_{uttid}"
        text = text.lower()
        token = "".join(token)

        res.append(f"{uttid} {start_text_token} {text}{enrollment_speech_token} 
                   {enroll_token}{generate_speech_token}{token}")

    print("Creating tts: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_se(
    root: Path,
    out_dir=Path("data"),
    enrollment_dict=enrollment_dict,
    start_text_token="<startoftext>",
    generate_speech_token="<generatespeech>",
    enrollment_speech_token="<enrollspeech>",
    use_cjk=True,
    use_enroll_speech=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token_source = read_text(root / f"token_source")
    uttid2token_target = read_text(root / f"token_target")
    
    res = []
    for uttid, text in uttid2text.items():
        token_source = uttid2token_source[uttid].split()
        token_target = uttid2token_target[uttid].split()
        
        if use_cjk:
            token_source = [unit2cjk(t) for t in token_source]
            token_target = [unit2cjk(t) for t in token_target]
            
        if use_enroll_speech:
            spk_id = uttid.split('_')[0]
            enroll_token = enrollspeech_dict[spk_id].split()
            if use_cjk:
                enroll_token = [unit2cjk(t) for t in enroll_token]
                enroll_token = ''.join(enroll_token)
            
        uttid = f"se_{uttid}"
        text = text.lower()
        token_source = "".join(token_source)
        token_target = "".join(token_target)

        res.append(f"{uttid} {start_speech_token}{token_source} {generate_text_token} {text} 
                   {enrollment_speech_token}{enroll_token}{generate_speech_token}{token_target}")

    print("Creating se: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_vc(
    root: Path,
    out_dir=Path("data"),
    enrollment_dict=enrollment_dict,
    start_text_token="<startoftext>",
    generate_speech_token="<generatespeech>",
    enrollment_speech_token="<enrollspeech>",
    use_cjk=True,
    use_enroll_speech=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token_source = read_text(root / f"token_source")
    uttid2token_target = read_text(root / f"token_target")
    
    res = []
    for uttid, text in uttid2text.items():
        token_source = uttid2token_source[uttid].split()
        token_target = uttid2token_target[uttid].split()
        
        if use_cjk:
            token_source = [unit2cjk(t) for t in token_source]
            token_target = [unit2cjk(t) for t in token_target]
            
        if use_enroll_speech:
            spk_id = uttid.split('_')[0]
            enroll_token = enrollspeech_dict[spk_id].split()
            if use_cjk:
                enroll_token = [unit2cjk(t) for t in enroll_token]
                enroll_token = ''.join(enroll_token)
            
        uttid = f"vc_{uttid}"
        text = text.lower()
        token_source = "".join(token_source)
        token_target = "".join(token_target)

        res.append(f"{uttid} {start_speech_token}{token_source} {generate_text_token} {text} 
                   {enrollment_speech_token}{enroll_token}{generate_speech_token}{token_target}")

    print("Creating vc: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return





def prepare_enroll_speech_dict(root: Path):
    enroll_dict = {}
    uttid2token = read_text(root)
    for uttid, token in uttid2text.items():
        spkid = utt_id.split("_")[0]
        token=token.split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]
        token = "".join(token)
        if spkid not in enroll_dict:
            enroll_dict[spkid]=tok
    print("Save enrollmet speech for ",len(enroll_dict.keys())," speakers.")
    return enroll_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help=" ", default="dump")
    parser.add_argument(
        "--start_text_token",
        type=str,
        help="Token to denote start of text as condition.",
        default="<startoftext>",
    )
    parser.add_argument(
        "--generate_text_token",
        type=str,
        help="Token to denote generate text.",
        default="<generatetext>",
    )
    parser.add_argument(
        "--start_speech_token",
        type=str,
        help="Token to denote start of speech as condition.",
        default="<startofspeech>",
    )
    parser.add_argument(
        "--generate_speech_token",
        type=str,
        help="Token to denote generate speech.",
        default="<generatespeech>",
    )
    parser.add_argument(
        "--enrollment_speech_token",
        type=str,
        help="Token to denote enrollment speech.",
        default="<enrollspeech>",
    )
    parser.add_argument(
        "--use_cjk",
        type=str2bool,
        help="Whether to map speech tokens into cjk. Needed for BPE training",
        default=True,
    )
    parser.add_argument(
        "--use_enroll_speech",
        type=str2bool,
        help="Whether to use enrollment speech",
        default=True,
    )

    args = parser.parse_args()
    out_dir = Path(args.path)

    # create empty file
    with (out_dir / "lm_text").open("w") as fp:
        print("Opened file:", out_dir)

    # prepare enrollment speech dict
    if use_enroll_speech:
        tts_enrollment_dict = prepare_enroll_speech_dict(out_dir / "speech/tts/token")
        vc_enrollment_dict = prepare_enroll_speech_dict(out_dir / "speech/vc/token_source")
        se_enrollment_dict = prepare_enroll_speech_dict(out_dir / "speech/se/token_source")
        
    
    
    # prepare textlm
    prepare_textlm(
        out_dir / "text/textlm",
        out_dir=out_dir,
        generate_text_token=args.generate_text_token,
    )

    # process speechlm
    prepare_speechlm(
        out_dir / "speech/speechlm",
        out_dir=out_dir,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
    )

    # process asr
    prepare_asr(
        out_dir / "speech/asr",
        out_dir=out_dir,
        start_speech_token=args.start_speech_token,
        generate_text_token=args.generate_text_token,
        use_cjk=args.use_cjk,
    )

    # process tts
    prepare_tts(
        out_dir / "speech/tts",
        out_dir=out_dir,
        enrollment_dict=tts_enrollment_dict,
        start_text_token=args.start_text_token,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
        use_enroll_speech=args.use_enroll_speech,
    )
    
    prepare_se(
        out_dir / "speech/se",
        out_dir=out_dir,
        enrollment_dict=se_enrollment_dict,
        start_text_token=args.start_text_token,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
        use_enroll_speech=args.use_enroll_speech,
    )
    
    prepare_vc(
        out_dir / "speech/vc",
        out_dir=out_dir,
        enrollment_dict=vc_enrollment_dict,
        start_text_token=args.start_text_token,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
        use_enroll_speech=args.use_enroll_speech,
    )

    with (Path("data") / "nlsyms.txt").open("w") as fp:
        fp.write(
            "{}\n{}\n{}\n{}\n{}\n".format(
                args.start_text_token,
                args.generate_text_token,
                args.start_speech_token,
                args.generate_speech_token,
                args.enrollment_speech_token,
            )
        )
