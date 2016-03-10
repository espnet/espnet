kaldi-io-for-python
===================
Helper IO functions to interface kaldi data types with python.
-------------------

** Supported data types **
- vector (integer)
- Vector (float,double)
- Matrix (float,double)
- Posterior (posteriors, nnet1 training targets, confusion networks, ...)

** Examples **

- Reading feature scp example:

  import kaldi_io
  for key,mat in kaldi_io.read_mat_scp(file):
    ...

- Writing feature ark to file/stream:

  import kaldi_io
  with open(ark_file,'w') as f:
    for key,vec in dict.iteritems(): 
      kaldi_io.write_mat(f, vec, key=key)

** License **
Apache License, Version 2.0 ('LICENSE-2.0.txt')

** Contact **
- If you have an extension to share, feel free to create a pull request.
- For feedback and suggestions you can create a GitHub 'Issue' in the project, 
  or contact me by email: vesis84@gmail.com
