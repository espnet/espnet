import os
import subprocess
import pandas as pd
from tqdm import tqdm
import argparse

def calc_CR(wav_scp, data_dir, res_dir):
    
    d = {"fname":[], "CR":[]}

    with open(wav_scp) as fo:
        commands = fo.readlines()
    
    for cmd in tqdm(commands):
        fname = cmd.split()[0]
        fpath = os.path.join(data_dir, "/".join(fname.split('_')[:-1]), fname)
        cmd_convert = cmd.split()[1:-2]
        #cmd_convert.pop(0)
        #cmd_convert.insert(0, 'ffmpeg')
        cmd_convert.pop(5)
        cmd_convert.insert(5, fpath+'.opus')
        cmd_convert.append(fpath+'.wav')
        cmd_convert = subprocess.run(cmd_convert, stdout=subprocess.PIPE, 
                                                text=True, check=True)
        print(cmd_convert)
        temp = subprocess.run(["gzip", "-k", fpath+'.wav'])
        fsize = subprocess.run(["du", fpath+'.wav'], stdout=subprocess.PIPE, 
                                            text=True, check=True)
        fsize_comp = subprocess.run(["du", fpath+'.wav'+'.gz'], stdout=subprocess.PIPE, 
                                            text=True, check=True)
        fsize = int(fsize.stdout.split('\t')[0])
        fsize_comp = int(fsize_comp.stdout.split('\t')[0])
        temp = subprocess.run(["rm", fpath+".wav"+".gz"])
        temp = subprocess.run(["rm", fpath+".wav"])
        CR = 1 - (fsize_comp/fsize)
        
        d["fname"].append(f)
        d["CR"].append(CR)

        df = pd.DataFrame.from_dict(d)
        df.to_csv(os.path.join(res_dir, 'compression_ratio.csv'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_scp', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')

    args = parser.parse_args()
    calc_CR(args.wav_scp, args.data_dir, args.res_dir)