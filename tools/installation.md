cupy installation failed, so I used `pip install cupy-cuda100` to install it manually, and set
`NO_CUPY` to 1 in `Makefile`

Other steps are the same.

```
. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py
[x] python=3.8.10 (default, May 19 2021, 18:05:58)  [GCC 7.3.0]

Python modules:
[x] torch=1.4.0
[ ] torch cuda
[x] torch cudnn=7603
[x] torch nccl
/home/storage15/tangjiyang/espnet/tools/anaconda/envs/espnet/lib/python3.8/site-packages/chainer/_environment_check.py:70: UserWarning: 
--------------------------------------------------------------------------------
CuPy (cupy-cuda100) version 9.1.0 may not be compatible with this version of Chainer.
Please consider installing the supported version by running:
  $ pip install 'cupy-cuda100>=6.0.0,<7.0.0'

  See the following page for more details:
    https://docs-cupy.chainer.org/en/latest/install.html
    --------------------------------------------------------------------------------

      warnings.warn(msg.format(
      [x] chainer=6.0.0
      [ ] chainer cuda
      [ ] chainer cudnn
      [ ] cupy
      [x] torchaudio=0.4.0
      [x] torch_optimizer=0.1.0
      [ ] warpctc_pytorch
      [ ] warprnnt_pytorch
      [ ] warp_rnnt
      [ ] chainer_ctc
      [ ] pyopenjtalk
      [ ] kenlm
      [ ] mmseg
      [x] espnet=0.9.9
      [ ] fairseq
      [ ] phonemizer
      [ ] gtn

      Executables:
      [x] sclite
      [x] sph2pipe
      [ ] PESQ
      [ ] BeamformIt

      INFO:
      Use 'installers/install_warp-ctc.sh' to install warpctc_pytorch
      Use 'installers/install_warp-transducer.sh' to install warprnnt_pytorch
      Use 'installers/install_warp-rnnt.sh' to install warp_rnnt
      Use 'installers/install_chainer_ctc.sh' to install chainer_ctc
      Use 'installers/install_pyopenjtalk.sh' to install pyopenjtalk
      Use 'installers/install_kenlm.sh' to install kenlm
      Use 'installers/install_py3mmseg.sh' to install mmseg
      Use 'installers/install_fairseq.sh' to install fairseq
      Use 'installers/install_phonemizer.sh' to install phonemizer
      Use 'installers/install_gtn.sh' to install gtn
      Use 'installers/install_pesq.sh' to install PESQ
      Use 'installers/install_beamformit.sh' to install BeamformIt
```
