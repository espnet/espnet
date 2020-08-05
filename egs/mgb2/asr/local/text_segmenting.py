#!/usr/bin/env python3

# Copyright 2020 Kanari AI (Amir Hussein)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pandas as pd
import argparse

def get_split(text, maxlen=200, overlap=50):
    """Segment text data with overlapped window of len maxlen"""
    l_total=[]
    l_partial=[]
    den = maxlen-overlap
    if len(text.split())//den > 0:
        n=len(text.split())//den
    else:
        n=1
    for w in range(n):
        if w == 0:
            l_partial =text.split()[:maxlen]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text.split()[w*den:w*den + maxlen]
            l_total.append(" ".join(l_partial))
    return l_total

def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_dir', help='Directory of text to be segmented', type=str, default='lm_text_clean_bw')
    return parser.parse_args()


def main():
	args = get_args()
	myfile = pd.read_csv(args.file_dir, header=None, encoding='utf-8')

	segmented_text=[]
	for i in range(len(myfile)):
		segmented_text.extend(get_split(myfile.iloc[i][0]))
		
	df=pd.DataFrame(segmented_text, index=None)
	df.to_csv("segmented_text", sep = '\n', header=False, index=False)
    
if __name__ == "__main__":
    main()
