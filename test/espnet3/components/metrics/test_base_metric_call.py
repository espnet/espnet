from espnet3.api.inference import Field
from espnet3.components.contract.metrics import check_metric_output
from espnet3.components.metrics.base_metric import BaseMetric


class Lengths(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("chars", "number"),)

    def __init__(self):
        super().__init__()
        self.add_state("chars", 0, "sum")
        self.add_state("calls", 0, "sum")

    def update(self, *, ref, hyp):
        self.chars += sum(len(r) for r in ref)
        self.calls += 1

    def compute(self):
        return {"chars": self.chars}

    def write_artifacts(self, test_dir):
        (test_dir / "lengths.txt").write_text(str(self.chars), encoding="utf-8")


ROWS = ["ab", "cde", "f"]


def _data(tmp_path):
    data = {}
    for name in ("ref", "hyp"):
        path = tmp_path / f"{name}.scp"
        lines = [f"utt{i} {row}" for i, row in enumerate(ROWS)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        data[name] = path
    return data


def test_call_over_scp_equals_in_memory_update(tmp_path):
    result = Lengths()(_data(tmp_path), "t", None)
    m = Lengths()
    m.update(ref=ROWS, hyp=ROWS)
    assert result == m.compute() == {"chars": 6}


def test_call_updates_in_chunks(tmp_path):
    m = Lengths()
    m.chunk_size = 2
    m(_data(tmp_path), "t", None)
    assert m.calls == 2


def test_call_resets_between_runs(tmp_path):
    m = Lengths()
    data = _data(tmp_path)
    m(data, "t", None)
    assert m(data, "t", None) == {"chars": 6}


def test_call_without_output_dir_writes_nothing(tmp_path):
    Lengths()(_data(tmp_path), "t", None)
    assert not (tmp_path / "t").exists()


def test_call_writes_artifacts_under_the_test_name(tmp_path):
    Lengths()(_data(tmp_path), "t", tmp_path / "out")
    assert (tmp_path / "out" / "t" / "lengths.txt").read_text(encoding="utf-8") == "6"


def test_compute_writes_no_file(tmp_path):
    m = Lengths()
    m.update(ref=ROWS, hyp=ROWS)
    before = sorted(tmp_path.rglob("*"))
    m.compute()
    assert sorted(tmp_path.rglob("*")) == before


def test_compute_result_passes_check_metric_output():
    m = Lengths()
    m.update(ref=ROWS, hyp=ROWS)
    check_metric_output(m, m.compute())
