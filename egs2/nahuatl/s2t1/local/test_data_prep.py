import os, subprocess, sys, tempfile, pytest

# Default to <repo_root>/hf_data (repo_root is 5 levels up from this file:
# local/ -> s2t1/ -> nahuatl/ -> egs2/ -> espnet/ -> <repo_root>).
_DEFAULT_HF_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../hf_data")
)
HF_DATA_DIR = os.environ.get("HF_DATA_DIR", _DEFAULT_HF_DATA_DIR)
SCRIPT = os.path.join(os.path.dirname(__file__), "data_prep.py")


def run_prep(tmpdir, split="hidalgo-train", token="<nah_hid>", max_examples=12):
    out_dir = os.path.join(tmpdir, "kaldi")
    wav_dir = os.path.join(tmpdir, "wav")
    result = subprocess.run(
        [
            sys.executable, SCRIPT,
            "--hf_data_dir", HF_DATA_DIR,
            "--split", split,
            "--output_dir", out_dir,
            "--wav_dir", wav_dir,
            "--region_token", token,
            "--max_examples", str(max_examples),
        ],
        capture_output=True, text=True,
    )
    return result, out_dir, wav_dir


def test_creates_all_four_kaldi_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result, out_dir, _ = run_prep(tmpdir)
        assert result.returncode == 0, result.stderr
        for fname in ("wav.scp", "text", "utt2spk", "spk2utt"):
            assert os.path.exists(os.path.join(out_dir, fname)), f"Missing {fname}"


def test_wav_scp_is_sox_pipe_with_16k_mono():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(tmpdir)
        lines = open(os.path.join(out_dir, "wav.scp")).readlines()
        assert len(lines) == 12
        utt_id, rest = lines[0].strip().split(" ", 1)
        assert rest.startswith("sox "), f"Expected sox pipe, got: {rest[:40]}"
        assert rest.endswith(" |"), f"Sox pipe must end with ' |': {rest[-10:]}"
        assert "-r 16000" in rest
        assert "-c 1" in rest
        assert "-t wav" in rest


def test_text_has_region_token_prefix():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(tmpdir, token="<nah_hid>")
        lines = open(os.path.join(out_dir, "text")).readlines()
        for line in lines:
            _, text = line.strip().split(" ", 1)
            assert text.startswith("<nah_hid><asr><notimestamps> "), (
                f"Bad text format: {text[:60]}"
            )


def test_utt2spk_spk2utt_are_consistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(tmpdir)
        utt2spk = {}
        for line in open(os.path.join(out_dir, "utt2spk")):
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
        spk2utt = {}
        for line in open(os.path.join(out_dir, "spk2utt")):
            parts = line.strip().split()
            spk2utt[parts[0]] = set(parts[1:])
        for utt, spk in utt2spk.items():
            assert spk in spk2utt, f"Speaker {spk} missing from spk2utt"
            assert utt in spk2utt[spk], f"{utt} missing from spk2utt[{spk}]"


def test_all_files_sorted_by_utt_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(tmpdir)
        for fname in ("wav.scp", "text", "utt2spk"):
            ids = [
                line.split()[0]
                for line in open(os.path.join(out_dir, fname))
            ]
            assert ids == sorted(ids), f"{fname} not sorted: {ids[:5]}"


def test_wav_files_are_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, wav_dir = run_prep(tmpdir)
        wav_files = os.listdir(wav_dir)
        assert len(wav_files) == 12
        assert all(f.endswith(".wav") for f in wav_files)
        # Each wav file must be > 0 bytes
        for f in wav_files:
            size = os.path.getsize(os.path.join(wav_dir, f))
            assert size > 44, f"{f} is suspiciously small ({size} bytes)"


def test_utt_ids_contain_no_special_chars():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(tmpdir)
        import re
        pat = re.compile(r'^[A-Za-z0-9_]+$')
        for line in open(os.path.join(out_dir, "wav.scp")):
            utt_id = line.split()[0]
            assert pat.match(utt_id), f"Bad utt_id: {utt_id}"


def test_ozg_region_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out_dir, _ = run_prep(
            tmpdir,
            split="orizaba-zongolica-train",
            token="<nah_ozg>",
        )
        lines = open(os.path.join(out_dir, "text")).readlines()
        _, text = lines[0].strip().split(" ", 1)
        assert text.startswith("<nah_ozg><asr><notimestamps> ")
