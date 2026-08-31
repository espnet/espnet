"""Optional branches of the conditional flow-matching wrapper.

``CFM.sample`` and ``CFM.forward`` accept several alternatives a recipe never
selects - text as raw strings, editing masks, guidance disabled, an inline
vocoder - which a normal train or infer run leaves untouched.
"""

import pytest
import torch

from espnet3.systems.tts.f5_tts.f5tts import F5TTS

MODEL_CONF = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    feed_forward_multiplier=1,
    text_embedding_size=16,
    convolution_layers=1,
    ode_solver_method="euler",
)
FEATS_CONF = dict(
    fs=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=100,
    mel_spec_type="vocos",
)


def _cfm(tokens):
    model = F5TTS(
        token_list=tokens,
        feats_extract_config=FEATS_CONF,
        **MODEL_CONF,
    )
    return model.cfm


@pytest.fixture
def cfm():
    return _cfm(["<blank>", "<unk>", "a", "b", "<sos/eos>"])


@pytest.fixture
def byte_cfm():
    """A vocab wide enough for F5's byte-level fallback tokenizer.

    Without a vocab_char_map, strings are tokenized to raw UTF-8 values, so the
    embedding table has to span the byte range.
    """
    return _cfm([f"<{i}>" for i in range(260)])


# ------------------------------------------------------------------- sampling


def test_text_may_be_given_as_raw_strings(byte_cfm):
    """Without a vocab map, strings fall back to F5's byte-level tokenizer."""
    assert byte_cfm.vocab_char_map is None

    out, _ = byte_cfm.sample(
        cond=torch.randn(1, 20, 100), text=["ab"], duration=30, steps=2
    )

    assert out.shape == (1, 30, 100)


def test_a_string_batch_must_match_the_condition_batch(cfm):
    with pytest.raises(AssertionError):
        cfm.sample(
            cond=torch.randn(1, 20, 100), text=["ab", "cd"], duration=30, steps=2
        )


def test_disabling_guidance_takes_the_single_pass_branch(cfm):
    """cfg_strength below 1e-5 skips the paired cond/uncond forward."""
    out, _ = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=2,
        cfg_strength=0.0,
    )

    assert out.shape == (1, 30, 100)
    assert torch.isfinite(out).all()


def test_no_ref_audio_blanks_the_conditioning(cfm):
    """Used to check how much the model leans on the prompt."""
    out, _ = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=2,
        no_ref_audio=True,
    )

    assert out.shape == (1, 30, 100)


def test_an_edit_mask_narrows_the_retained_condition(cfm):
    """Speech editing keeps only the frames the mask marks."""
    edit_mask = torch.zeros(1, 20, dtype=torch.bool)
    edit_mask[:, :10] = True

    out, _ = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=2,
        lens=torch.tensor([20]),
        edit_mask=edit_mask,
    )

    assert out.shape == (1, 30, 100)


def test_duplicate_test_starts_the_trajectory_partway(cfm):
    """An inner-timestep diagnostic: start from t_inter, not from noise."""
    out, trajectory = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=4,
        duplicate_test=True,
        t_inter=0.5,
    )

    assert out.shape == (1, 30, 100)
    # steps is scaled by (1 - t_start), so the grid is shorter.
    assert trajectory.shape[0] < 5


def test_a_batch_larger_than_one_builds_an_attention_mask(cfm):
    """Single-sample inference skips the mask; a real batch cannot."""
    out, _ = cfm.sample(
        cond=torch.randn(2, 20, 100),
        text=torch.tensor([[2, 3], [3, 2]]),
        duration=torch.tensor([30, 26]),
        steps=2,
    )

    assert out.shape == (2, 30, 100)


def test_an_inline_vocoder_is_applied_to_the_output(cfm):
    """sample() can hand back audio directly instead of mel."""

    def vocoder(mel):  # [b, d, n] -> [b, nw]
        return torch.zeros(mel.shape[0], mel.shape[-1] * 256)

    out, _ = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=2,
        vocoder=vocoder,
    )

    assert out.shape == (1, 30 * 256)


def test_epss_is_skipped_when_it_has_no_schedule_for_the_step_count(cfm):
    """Only some step counts have a pruned grid; the rest use linspace."""
    out, trajectory = cfm.sample(
        cond=torch.randn(1, 20, 100),
        text=torch.tensor([[2, 3]]),
        duration=30,
        steps=7,
        use_epss=True,
    )

    assert out.shape == (1, 30, 100)
    assert trajectory.shape[0] == 8


# -------------------------------------------------------------------- forward


def test_forward_accepts_a_raw_waveform(cfm):
    """ndim == 2 means audio, so CFM extracts the mel itself."""
    loss, cond, pred = cfm(torch.randn(1, 24000), text=torch.tensor([[2, 3]]))

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert cond.shape[-1] == pred.shape[-1] == 100


def test_forward_accepts_text_as_strings(byte_cfm):
    loss, _, _ = byte_cfm(torch.randn(1, 40, 100), text=["ab"])

    assert torch.isfinite(loss)


def test_forward_defaults_lens_to_the_full_sequence(cfm):
    """The trainer normally supplies lens from the collate function."""
    with_lens, _, _ = cfm(
        torch.randn(1, 40, 100), text=torch.tensor([[2, 3]]), lens=torch.tensor([40])
    )
    without_lens, _, _ = cfm(torch.randn(1, 40, 100), text=torch.tensor([[2, 3]]))

    assert torch.isfinite(with_lens) and torch.isfinite(without_lens)
