#!/usr/bin/env python

import numpy as np
import kaldi_io

print('testing int32-vector i/o')
i32_vec = { k:v for k,v in kaldi_io.read_vec_int_ark('data/ali.ark') } # binary,
i32_vec2 = { k:v for k,v in kaldi_io.read_vec_int_ark('data/ali_ascii.ark') } # ascii,
# - store,
with kaldi_io.open_or_fd('data_re-saved/ali.ark','wb') as f:
  for k,v in i32_vec.items(): kaldi_io.write_vec_int(f, v, k)
# - read and compare,
for k,v in kaldi_io.read_vec_int_ark('data_re-saved/ali.ark'):
  assert(np.array_equal(v,i32_vec[k]))

print('testing float-vector i/o')
flt_vec = { k:v for k,v in kaldi_io.read_vec_flt_scp('data/conf.scp') } # scp,
flt_vec2 = { k:v for k,v in kaldi_io.read_vec_flt_ark('data/conf.ark') } # binary-ark,
flt_vec3 = { k:v for k,v in kaldi_io.read_vec_flt_ark('data/conf_ascii.ark') } # ascii-ark,
# - store,
with kaldi_io.open_or_fd('data_re-saved/conf.ark','wb') as f:
  for k,v in flt_vec.items(): kaldi_io.write_vec_flt(f, v, k)
# - read and compare,
for k,v in kaldi_io.read_vec_flt_ark('data_re-saved/conf.ark'):
  assert(np.array_equal(v,flt_vec[k]))

print('testing matrix i/o')
flt_mat = { k:m for k,m in kaldi_io.read_mat_scp('data/feats_ascii.scp') } # ascii-scp,
flt_mat2 = { k:m for k,m in kaldi_io.read_mat_ark('data/feats_ascii.ark') } # ascii-ark,
flt_mat3 = { k:m for k,m in kaldi_io.read_mat_ark('data/feats.ark') } # ascii-ark,
# - store,
with kaldi_io.open_or_fd('data_re-saved/mat.ark','wb') as f:
  for k,m in flt_mat3.items(): kaldi_io.write_mat(f, m, k)
# - read and compare,
for k,m in kaldi_io.read_mat_ark('data_re-saved/mat.ark'):
  assert(np.array_equal(m, flt_mat3[k]))

print('testing i/o with pipes')
flt_mat4 = { k:m for k,m in kaldi_io.read_mat_ark('ark:copy-feats ark:data/feats.ark ark:- |') }
# - store,
with kaldi_io.open_or_fd('ark:| copy-feats ark:- ark:data_re-saved/mat_pipe.ark','wb') as f:
  for k,m in flt_mat4.items(): kaldi_io.write_mat(f, m, k)
# - read and compare,
for k,m in kaldi_io.read_mat_ark('data_re-saved/mat_pipe.ark'):
  assert(np.array_equal(m, flt_mat4[k]))

i32_vec3 = { k:v for k,v in kaldi_io.read_vec_int_ark('ark:copy-int-vector ark:data/ali.ark ark:- |') }
flt_vec4 = { k:v for k,v in kaldi_io.read_vec_flt_ark('ark:copy-vector ark:data/conf.ark ark:- |') }

print('all tests passed...')
