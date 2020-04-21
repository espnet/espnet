import pytest

from espnet.nets.asr_interface import dynamic_import_asr


@pytest.mark.parametrize(
    "name, backend",
    [(nn, backend) for nn in ("transformer", "rnn") for backend in ("pytorch",)],
)
def test_asr_build(name, backend):
    model = dynamic_import_asr(name, backend).build(
        10, 10, mtlalpha=0.123, adim=4, eunits=3, dunits=3, elayers=2, dlayers=2
    )
    assert model.mtlalpha == 0.123
