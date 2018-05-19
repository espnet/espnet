#!/bin/env python

# Test reading a 'confusion network', represented by same Kaldi type as Posterior,

import kaldi_io

cnet_f='data/1.cnet'
cntime_f='data/1.cntime'

cnet = [ (k,v) for k,v in kaldi_io.read_cnet_ark(cnet_f) ]
cntime = [ (k,v) for k,v in kaldi_io.read_cntime_ark(cntime_f) ]

