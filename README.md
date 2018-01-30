kaldi-io-for-python
===================
``Glue'' code connecting kaldi data and python.
-------------------

#### Supported data types
- vector (integer)
- Vector (float, double)
- Matrix (float, double)
- Posterior (posteriors, nnet1 training targets, confusion networks, ...)

#### Examples

###### Reading feature scp example:
```python
import kaldi_io
for key,mat in kaldi_io.read_mat_scp(file):
  ...
```

###### Writing feature ark to file/stream:
```python
import kaldi_io
with open(ark_file,'wb') as f:
  for key,mat in dict.iteritems(): 
    kaldi_io.write_mat(f, mat, key=key)
```

###### Writing features as 'ark,scp' by pipeline with 'copy-feats':
```python
import kaldi_io
ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:data/feats2.ark,data/feats2.scp'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
  for key,mat in dict.iteritems(): 
    kaldi_io.write_mat(f, mat, key=key)
```


#### Install
- run `git clone https://github.com/vesis84/kaldi-io-for-python.git <kaldi-io-dir>`
- add `PYTHONPATH=${PYTHONPATH}:<kaldi-io-dir>` to `$HOME/.bashrc`
- now the `import kaldi_io` will work from any location


#### License
Apache License, Version 2.0 ('LICENSE-2.0.txt')

#### Contact
- If you have an extension to share, please create a pull request.
- For feedback and suggestions, please create a GitHub 'Issue' in the project.
- For the positive reactions =) I am also reachable by email: vesis84@gmail.com
