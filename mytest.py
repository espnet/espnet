from argparse import Namespace

from test.test_batch_beam_search import test_batch_beam_search_equal
from test.test_beam_search import transformer_args
import logging


lstm_lm = Namespace(type="lstm", layer=1, unit=2, dropout_rate=0.0)
gru_lm = Namespace(type="gru", layer=1, unit=2, dropout_rate=0.0)
transformer_lm = Namespace(
    layer=1, unit=2, att_unit=2, embed_unit=2, head=1, pos_enc="none", dropout_rate=0.0
)

args=[
    (nn, args, ctc, lm_nn, lm_args, lm, ngram, bonus, device, dtype)
    for device in ("cpu", "cuda")
    # (("rnn", rnn_args),)
    for nn, args in (("transformer", transformer_args),)
    for ctc in (0.0,)  # 0.5, 1.0)
    for lm_nn, lm_args in (
        ("default", lstm_lm),
        ("default", gru_lm),    
        ("transformer", transformer_lm),
    )
    for lm in (0.0, 0.5)
    for ngram in (0.0, 0.5)
    for bonus in (0.0, 0.1)
    for dtype in ("float32", "float64")  # TODO(karita): float16
]

logging.basicConfig(level=logging.DEBUG)
for arg in args:
    test_batch_beam_search_equal(*arg)
