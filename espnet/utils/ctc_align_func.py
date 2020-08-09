import json
import logging

import torch
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.align_core import align_run
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets


def ctc_align(args):
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Decoding device={device}")
    model.to(device=device).eval()

    # read json data
    with open(args.data_json, "rb") as f:
        js = json.load(f)["utts"]

    with open(args.utt_text, "r") as f:
        lines = f.readlines()
        i = 0
        text = {}
        segment_names = {}
        for name in js.keys():
            text_per_audio = []
            segment_names_per_audio = []
            while i < len(lines) and lines[i].startswith(name):
                text_per_audio.append(lines[i][lines[i].find(" ") + 1:])
                segment_names_per_audio.append(lines[i][:lines[i].find(" ")])
                i += 1
            text[name] = text_per_audio
            segment_names[name] = segment_names_per_audio

    with torch.no_grad():
        new_js = {}
        with open(args.output, "w") as f:
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) aligning " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat, label = load_inputs_and_targets(batch)
                feat = feat[0]
                enc_output = model.encode(torch.as_tensor(feat).to(device)).unsqueeze(0)
                # Apply ctc part
                lpz = model.ctc.log_softmax(enc_output)[0].cpu().numpy()

                boundaries = align_run(lpz, text[name], train_args.char_list)

                for i, boundary in enumerate(boundaries):
                    f.write(segment_names[name][i] + " " + name + " " + str(boundary[0]) + " " + str(boundary[1]) + " " + str(boundary[2]) + "\n")
                
