"""Metrics for OWSM: the three error rates ``egs2/TEMPLATE/s2t1/s2t.sh`` reports.

Stage 13 of ``egs2/TEMPLATE/s2t1/s2t.sh`` is the whole of OWSM's evaluation: it
tokenizes to char, word and BPE and runs sclite over each, giving ``cer``,
``wer`` and ``ter``. :class:`~espnet3.systems.esp2_s2t.metrics.cer.CER`,
:class:`~espnet3.systems.esp2_s2t.metrics.wer.WER` and
:class:`~espnet3.systems.esp2_s2t.metrics.ter.TER` reproduce it with jiwer in
place of sclite, scoring every row: s2t.sh does not split by task, so a
translation is scored against its translation as if it were a transcript.

No BLEU. s2t.sh computes none, and no owsm recipe adds one, so a BLEU here
would be a number with nothing to compare against.

Two preprocessing details are reproduced because they change the numbers:

* Tags are removed from both sides, using the recipe's own ``nlsyms``
  inventory so that only symbols the data can actually contain are stripped.
  s2t.sh asks for this and cannot achieve it on OWSM text.

* No text cleaner by default. ``ref_cleaner`` and ``hyp_cleaner`` are separate
  in s2t.sh and both default to ``none``. ``egs2/owsm_v3/s2t1/run.sh:18``
  carries ``# --cleaner whisper_en --hyp_cleaner whisper_en`` commented out;
  pass ``ref_cleaner: [whisper_en]`` to turn it on.
"""
