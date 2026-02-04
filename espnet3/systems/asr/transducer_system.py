"""ASR transducer system wrapper."""

from espnet3.systems.asr.system import ASRSystem


class ASRTransducerSystem(ASRSystem):
    """ASR Transducer-specific system.

    This system adds:
      - Tokenizer training inside train()
    """

    pass
