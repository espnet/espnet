import os
import subprocess
import pandas as pd
from tqdm import tqdm
import argparse

def calc_CR(data_dir, res_dir):
    '''
    Params
        data_dir (str): directory where the audio files are stored.
        res_dir (str): directory where the csv with compression ratios will be stored.
    '''
    d = {"fname":[], "CR":[]}
    
    fnames = os.listdir(data_dir)
    for f in tqdm(fnames):
        fpath = os.path.join(data_dir, f)
        temp = subprocess.run(["gzip", "-k", fpath])
        fsize = subprocess.run(["du", fpath], stdout=subprocess.PIPE, 
                                              text=True, check=True)
        fsize_comp = subprocess.run(["du", fpath+'.gz'], stdout=subprocess.PIPE, 
                                              text=True, check=True)
        fsize = int(fsize.stdout.split('\t')[0])
        fsize_comp = int(fsize_comp.stdout.split('\t')[0])
        temp = subprocess.run(["rm", fpath+".gz"])
        CR = 1 - (fsize_comp/fsize)
        
        #Calculate CR and store
        d["fname"].append(f)
        d["CR"].append(CR)
        
        df = pd.DataFrame.from_dict(d)
        df.to_csv(os.path.join(res_dir, 'compression_ratio.csv'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dir.')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Path to dir where csv with the results will be stored.')

    args = parser.parse_args()
    calc_CR(args.data_dir, args.res_dir)