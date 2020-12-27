import os, sys
from importlib import import_module
sys.path.insert(0,os.getcwd())

import numpy as np
import kaldiio

import torch
from local.CDVQVAE.cdvqvae import Model


def quantize(model_config, model_path, utt2path, output_file):
    model = Model(model_config)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()

    for utt, path in utt2path.items():
        x = torch.from_numpy(kaldiio.load_mat(path)).cuda()
        x = x.unsqueeze(0).transpose(1, 2) # Size: (1, 8192, t)
        with torch.no_grad():
            idx = model.quantize(x).squeeze() # Size: (t)
        idx = idx.int()

        output_file.write(utt)
        for i in range(len(idx)):
            output_file.write(' '+ str(idx[i].item()))
        output_file.write('\n')
        
if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p','--model_path', type=str,
                        help='Path to checkpoint with model')
    parser.add_argument('-f','--input_file', type=str,
                        help='The feats.csp file containing the lip features')
    parser.add_argument('-o','--output_file_path', type=str,
                        help='Path to the output text file')


    args = parser.parse_args()
    
    # Load Config.
    with open(args.config) as f:
        data = f.read()
    model_config = json.loads(data)
    model_path = args.model_path

    # Load wav.scp & spk2spk_id
    with open(args.input_file) as rf:
        utt2path = dict([line.rstrip().split() for line in rf.readlines()])
    
    output_file = open(args.output_file_path, 'w')
    quantize(model_config, model_path, utt2path, output_file)
