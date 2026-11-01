from argparse import Namespace

import pytest

from espnet2.s2t.preprocessor_conf import train_preprocessor_conf
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.s2t_ctc import S2TTask as S2TCTCTask


@pytest.mark.parametrize(
    "task, preprocessor",
    [(S2TTask, "s2t"), (S2TCTCTask, "s2t"), (S2TCTCTask, "s2t_ctc")],
)
def test_fills_what_the_config_left_at_the_default(task, preprocessor):
    # the OWSM-CTC configs set speech_length and leave fs unset
    args = Namespace(preprocessor=preprocessor, preprocessor_conf={"speech_length": 30})
    conf = train_preprocessor_conf(task, args)
    assert conf["fs"] == 16000
    assert conf["speech_length"] == 30


def test_the_config_wins_over_the_default():
    args = Namespace(preprocessor="s2t_ctc", preprocessor_conf={"fs": 8000})
    assert train_preprocessor_conf(S2TCTCTask, args)["fs"] == 8000


def test_does_not_change_the_training_args():
    stored = {"speech_length": 30}
    train_preprocessor_conf(
        S2TCTCTask, Namespace(preprocessor="s2t_ctc", preprocessor_conf=stored)
    )
    assert stored == {"speech_length": 30}


def test_without_a_preprocessor_returns_the_config_as_it_is():
    args = Namespace(preprocessor_conf={"speech_length": 30})
    assert train_preprocessor_conf(S2TCTCTask, args) == {"speech_length": 30}
