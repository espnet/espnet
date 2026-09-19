"""kNN-VC components: WavLM encoder, kNN matcher, HiFi-GAN vocoder, inference model.

ESPnet3 integration of kNN-VC by Matthew Baas, Benjamin van Niekerk and Herman
Kamper, "Voice Conversion With Just Nearest Neighbors", Interspeech 2023
(https://arxiv.org/abs/2305.18975). The method, training recipe, HiFi-GAN
configuration and released checkpoints are theirs (https://github.com/bshall/knn-vc,
MIT License); this package only adapts them to the ESPnet3 stage pipeline.
"""
