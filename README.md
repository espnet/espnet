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

#### License
Apache License, Version 2.0 ('LICENSE-2.0.txt')

#### Contact
- If you have an extension to share, please create a pull request.
- For feedback and suggestions, please create a GitHub 'Issue' in the project.
- For the positive reactions =) I am also reachable by email: vesis84@gmail.com
