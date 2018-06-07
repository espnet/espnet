#!/usr/bin/env python

# Copyright 2012  Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# ./gen_hamm_mat.py
# script generates diagonal matrix with hamming window values

from math import *
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--fea-dim', dest='dim', help='feature dimension')
parser.add_option('--splice', dest='splice', help='applied splice value')
(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)

dim=int(options.dim)
splice=int(options.splice)


#generate the diagonal matrix with hammings
M_2PI = 6.283185307179586476925286766559005

dim_mat=(2*splice+1)*dim
timeContext=2*splice+1
print '['
for row in range(dim_mat):
    for col in range(dim_mat):
        if col!=row:
            print '0',
        else:
            i=int(row/dim)
            print str(0.54 - 0.46*cos((M_2PI * i) / (timeContext-1))),
    print

print ']'


