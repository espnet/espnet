"""LACE (Layer-Adaptive Codec Encoding): dynamic frame rate codec for ESPnet3.

Wraps the residual vector quantizer of an espnet2 GAN codec (SoundStream,
Encodec, DAC, FunCodec) so that every codebook layer quantizes
segment-averaged frames under its own compression rate. The model code lives
under ``src/``:

- ``compression``: the shared ``BaseCompressionModel`` and its
  ``CompressionOutput`` record.
- ``cosine_similarity``, ``density_peak``, ``dp_segmentation``,
  ``identity``: the compression models.
- ``quantizer``: the quantizer wrappers that call a compression model
  before every codebook layer.
- ``lace``: the ``LACE`` model itself (an espnet2 GAN codec with the
  wrapped quantizer), which training configs reach through
  ``model._target_`` with ``task:`` left unset.
- ``gan_lightning_module``, ``gan_trainer``: the generator/discriminator
  training loop the wrapped codec (an ``AbsGANESPnetModel``) needs.

The two stage-facing modules sit at the package root:

- ``system``: ``LACESystem``, the staged pipeline a recipe under
  ``egs3/<corpus_name>/lace/`` runs; it builds the GAN trainer and writes
  the ``lace_model.yaml`` spec at training time.
- ``inference``: ``Inference``, the ``model`` of an inference config
  (``espnet3.api.inference``), wrapping ``LACEAudioCoding``, the encode ->
  decode roundtrip of a fine-tuned codec at a chosen compression rate.

The model is described in:

Thanapat Trachu, Samuele Cornell, William Chen, Shinji Watanabe.
"LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs"
SLT 2026. https://arxiv.org/abs/2609.17509

If you use this model in your research, please cite the paper above.
Ported from https://github.com/NewGamezzz/Multi-Compression-Audio-Codec.
"""
