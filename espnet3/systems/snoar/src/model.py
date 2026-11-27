"""Training objectives: supervised seed, DNF, MixIT, SNOAR and RemixIT.

``SNOARModel`` is the model the ESPnet3 Lightning wrapper trains. Its
``forward(**batch)`` returns ``(loss, stats, weight)`` like any ESPnet model,
so the stock ``train`` stage, checkpointing and resume apply unchanged.

The SNOAR update is the reference's ``bdnf_cyc`` arm (``run_whamr_compare.py``
at bDNF 2b1e186), step for step:

1. The EMA teacher reads the un-augmented recording ``x``: speech ``sp`` and
   noise ``nd`` (:func:`~espnet3.systems.snoar.src.readout.read_teacher`).
2. Observation adding: ``tgt = (1 - gamma) sp + gamma x``, ``gamma``
   cosine-annealed over generations.
3. Two cross-example remixes ``tgt + nd[perm]`` with the donor noise scaled
   to an SNR drawn from the teacher's own running estimate of the input SNR
   (``10 log10 ||sp||^2 / ||x - sp||^2``, plus 1.5 dB jitter).
4. Loss: SI-SDR on speech and noise for each view, plus ``cycle_weight``
   times the SI-SDR between the two views' speech estimates.
5. After every optimizer step the teacher takes an EMA step
   (:class:`~espnet3.systems.snoar.src.callbacks.TeacherEMACallback`).

Nothing in the SNOAR and RemixIT paths reads a reference signal. If the batch
carries ``speech_ref``, validation reports SI-SDR against it as a diagnostic.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from espnet3.systems.snoar.src.net import (
    EnhNet,
    build_net,
    load_net,
    resolve_model_dir,
)
from espnet3.systems.snoar.src.ops import (
    add_observation,
    anneal_gamma,
    compute_mixit_loss,
    compute_si_sdr,
    compute_sisdr_loss,
    normalize_gain,
    pair_by_roll,
    scale_to_snr,
)
from espnet3.systems.snoar.src.readout import (
    check_readout,
    read_speech,
    read_student,
    read_teacher,
)

logger = logging.getLogger(__name__)

MODES = ("supervised", "dnf", "mixit", "snoar", "remixit")
_SELF_TRAINING = ("snoar", "remixit")
_MODE_READOUT = {"dnf": "dnf", "mixit": "mixit", "supervised": "raw"}


class SNOARModel(nn.Module):
    """Train a speech enhancer with one of :data:`MODES`.

    Args:
        mode: ``supervised`` (references ``speech_ref``/``noise_ref``), ``dnf``
            or ``mixit`` (noisy ``speech_mix`` plus noise-only ``noise2``), or
            ``snoar``/``remixit`` (noisy ``speech_mix`` only, plus a seed).
        net: ESPnet2 enhancement config of a freshly initialized network.
        init: Directory written by the ``export`` stage or an ESPnet2 model
            tag to start from. Required for ``snoar`` and ``remixit``.
        readout: How the seed's outputs are read (``dnf``, ``mixit``,
            ``raw``). Defaults to the seed's ``snoar.yaml``, else to the
            mode's own readout, else ``raw``.
        mix_consistency: Overrides the network's mixture-consistency mode.
        fs: Sampling rate of the audio, passed to the network's encoder.
            STFT encoders with ``default_fs`` (e.g. the URGENT 2025
            TF-GridNet, ``default_fs: 8000``) scale their window to it.
        tau_rel: Guard of the DNF projection.
        gamma: Observation-adding weight at generation 1.
        gamma_end: Observation-adding weight at the last generation.
        n_generations: Number of generations of the anneal.
        steps_per_generation: Optimizer steps per generation.
        ema_decay: Teacher retention per optimizer step.
        views: 1 or 2 remix views per step (2 enables the cycle term).
        cycle_weight: Weight of the cross-view consistency term.
        remix_level: ``snr`` (self-SNR matched), ``median`` (bank-median noise
            energy) or ``raw`` (the donor's own level).
        purify_noise: ``decontaminate`` the teacher noise (``dnf``/``mixit``).
        snr_jitter: Standard deviation (dB) added to bank SNR draws.
        bank_size: Capacity of the self-estimated SNR / energy banks.
        scale_w: Scale-anchor weight of the SI-SDR losses (reference: 0).
        valid_utmos: Also report the UTMOS (UTMOS22 strong, SpeechMOS) of
            the exported estimate and of the input at validation: a
            reference-free diagnostic for corpora without clean speech.

    Raises:
        ValueError: For an unknown mode, a missing seed, or a readout that
            does not fit the network.
    """

    def __init__(
        self,
        mode: str,
        net: Optional[Dict[str, Any]] = None,
        init: Optional[str] = None,
        readout: Optional[str] = None,
        mix_consistency: Optional[str] = None,
        fs: Optional[int] = 16000,
        tau_rel: float = 1e-3,
        gamma: float = 0.2,
        gamma_end: float = 0.05,
        n_generations: int = 16,
        steps_per_generation: int = 5000,
        ema_decay: float = 0.999,
        views: int = 2,
        cycle_weight: float = 0.3,
        remix_level: str = "snr",
        purify_noise: bool = True,
        snr_jitter: float = 1.5,
        bank_size: int = 4000,
        scale_w: float = 0.0,
        valid_utmos: bool = False,
    ) -> None:
        """Build the network (and the teacher for self-training modes)."""
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if views not in (1, 2):
            raise ValueError(f"views={views}: expected 1 or 2")
        if remix_level not in ("snr", "median", "raw"):
            raise ValueError(f"remix_level {remix_level!r} not in snr|median|raw")
        if (net is None) == (init is None):
            raise ValueError("give exactly one of `net` (new network) or `init` (seed)")
        if mode in _SELF_TRAINING and init is None:
            raise ValueError(
                f"mode {mode!r} needs a seed: set `init` to an exported model "
                "directory (DNF, MixIT or supervised) or an ESPnet2 model tag"
            )
        self.mode = mode
        if init is not None:
            init = resolve_model_dir(str(init))
            self.net = load_net(init, mix_consistency=mix_consistency, fs=fs)
            stored = _read_sidecar(init).get("readout")
        else:
            self.net = build_net(net, mix_consistency=mix_consistency or "none", fs=fs)
            stored = None
        self.readout = readout or stored or _MODE_READOUT.get(mode, "raw")
        check_readout(self.readout, self.net.num_outputs)
        _check_mode_slots(mode, self.net)

        self.tau_rel = tau_rel
        self.gamma, self.gamma_end = gamma, gamma_end
        self.n_generations = n_generations
        self.steps_per_generation = steps_per_generation
        self.ema_decay = ema_decay
        self.views, self.cycle_weight = views, cycle_weight
        self.remix_level = remix_level
        self.purify_noise = purify_noise
        self.snr_jitter = snr_jitter
        self.scale_w = scale_w
        self.fs = int(fs or 16000)
        self.valid_utmos = valid_utmos
        self._utmos = None  # loaded on first use, see _diagnose_utmos
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

        self.teacher: Optional[EnhNet] = None
        if mode in _SELF_TRAINING:
            self.teacher = copy.deepcopy(self.net)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()
            # Fixed-size rings, so they live in the state dict and resume.
            self.register_buffer("snr_bank", torch.zeros(bank_size))
            self.register_buffer("nrg_bank", torch.zeros(bank_size))
            self.register_buffer("bank_count", torch.zeros((), dtype=torch.long))
        logger.info(
            "SNOARModel mode=%s readout=%s outputs=%d",
            mode,
            self.readout,
            self.net.num_outputs,
        )

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True) -> "SNOARModel":
        """Set train/eval mode; the teacher always stays in eval mode."""
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    @property
    def generation(self) -> int:
        """Return the current SNOAR generation, counted from 1."""
        return int(self.num_updates) // max(1, self.steps_per_generation) + 1

    @property
    def current_gamma(self) -> float:
        """Return the observation-adding weight for the current generation."""
        return anneal_gamma(
            self.generation, self.n_generations, self.gamma, self.gamma_end
        )

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Take one EMA step of the teacher towards the student."""
        self.num_updates += 1
        if self.teacher is None:
            return
        d = self.ema_decay
        for pt, ps in zip(self.teacher.parameters(), self.net.parameters()):
            pt.mul_(d).add_(ps.detach().to(pt.dtype), alpha=1.0 - d)
        for bt, bs in zip(self.teacher.buffers(), self.net.buffers()):
            bt.copy_(bs)

    def inference_net(self) -> EnhNet:
        """Return the network to export: the EMA teacher when there is one."""
        return self.teacher if self.teacher is not None else self.net

    # ------------------------------------------------------------------ #
    def forward(
        self,
        speech_mix: Optional[torch.Tensor] = None,
        noise2: Optional[torch.Tensor] = None,
        speech_ref: Optional[torch.Tensor] = None,
        noise_ref: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Return ``(loss, stats, weight)`` for one batch.

        Args:
            speech_mix: Noisy recordings ``(B, T)``.
            noise2: Noise-only signals ``(B, T)`` (``dnf``, ``mixit``).
            speech_ref: Reverberant speech reference (``supervised``; read by
                every other mode only for the validation diagnostic).
            noise_ref: Noise reference (``supervised``).
            **kwargs: Ignored (``*_lengths`` from the collator).
        """
        if speech_mix is None:
            raise ValueError("batch has no `speech_mix`")
        x = speech_mix.float()
        if self.mode == "supervised":
            loss, stats = self._supervised(x, speech_ref, noise_ref)
        elif self.mode == "dnf":
            loss, stats = self._dnf(x, _require(noise2, "noise2", "dnf"))
        elif self.mode == "mixit":
            loss, stats = self._mixit(x, _require(noise2, "noise2", "mixit"))
        else:
            loss, stats = self._self_train(x)
        if not self.training and speech_ref is not None:
            stats.update(self._diagnose(x, speech_ref.float()))
        if not self.training and self.valid_utmos:
            stats.update(self._diagnose_utmos(x))
        stats["loss"] = loss.detach()
        weight = torch.tensor(x.shape[0], device=x.device)
        return loss, stats, weight

    def _call(self, net: EnhNet, x: torch.Tensor) -> torch.Tensor:
        return net(x).float()

    def _supervised(self, x, speech_ref, noise_ref):
        s = _require(speech_ref, "speech_ref", "supervised").float()
        out = self._call(self.net, x)
        loss = compute_sisdr_loss(out[:, 0], s, self.scale_w)
        if out.shape[1] == 2:
            n = _require(noise_ref, "noise_ref", "supervised (2 slots)").float()
            loss = loss + compute_sisdr_loss(out[:, 1], n, self.scale_w)
        return loss, {}

    def _dnf(self, x1, n2):
        n2 = n2.float()
        out = self._call(self.net, x1 + n2)
        loss = compute_sisdr_loss(out[:, 0], x1, self.scale_w) + compute_sisdr_loss(
            out[:, 1], n2, self.scale_w
        )
        return loss, {}

    def _mixit(self, m1, m2):
        m2 = m2.float()
        out = self._call(self.net, m1 + m2)
        return compute_mixit_loss(out[:, 0], out[:, 1], out[:, 2], m1, m2), {}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_banks(self, x, sp, nd):
        resid = x - sp
        snr = 10 * torch.log10(
            (sp**2).sum(-1).clamp_min(1e-8) / (resid**2).sum(-1).clamp_min(1e-8)
        )
        nrg = (nd**2).sum(-1)
        cap = self.snr_bank.numel()
        idx = (self.bank_count + torch.arange(len(snr), device=x.device)) % cap
        self.snr_bank[idx] = snr.to(self.snr_bank.dtype)
        self.nrg_bank[idx] = nrg.to(self.nrg_bank.dtype)
        self.bank_count += len(snr)

    def _bank(self, bank: torch.Tensor) -> torch.Tensor:
        return bank[: int(min(int(self.bank_count), bank.numel()))]

    def _level(self, tgt, donor):
        """Level the permuted donor noise (``BDNF_REMIXLVL``)."""
        if self.remix_level == "raw":
            return donor
        if self.remix_level == "snr":
            bank = self._bank(self.snr_bank)
            draw = bank[torch.randint(0, len(bank), (tgt.shape[0],), device=tgt.device)]
            draw = draw + self.snr_jitter * torch.randn_like(draw)
            return scale_to_snr(tgt, donor, draw)
        return normalize_gain(donor, self._bank(self.nrg_bank).median())

    def _self_train(self, x):
        with torch.no_grad():
            sp, nd = read_teacher(
                self._call(self.teacher, x),
                x,
                self.readout,
                self.tau_rel,
                self.purify_noise,
            )
            if self.training:
                self._update_banks(x, sp, nd)
        B, dev = x.shape[0], x.device
        if self.mode == "remixit":
            perm = pair_by_roll(B, dev, max(1, B // 2))
            tgt_n = (x - sp)[perm]
            sh, nh = read_student(self._call(self.net, sp + tgt_n), sp + tgt_n)
            loss = compute_sisdr_loss(sh, sp, self.scale_w) + compute_sisdr_loss(
                nh, tgt_n, self.scale_w
            )
            return loss, {}
        gamma = self.current_gamma
        tgt = add_observation(sp, x, gamma)
        with torch.no_grad():
            if not self.training and int(self.bank_count) == 0:
                self._update_banks(x, sp, nd)  # validation before any training step
            n1 = self._level(tgt, nd[pair_by_roll(B, dev, max(1, B // 2))])
            n2 = self._level(tgt, nd[pair_by_roll(B, dev, max(1, B // 3))])
        sh1, nh1 = read_student(self._call(self.net, tgt + n1), tgt + n1)
        loss = compute_sisdr_loss(sh1, tgt, self.scale_w) + compute_sisdr_loss(
            nh1, n1, self.scale_w
        )
        if self.views == 2:
            sh2, nh2 = read_student(self._call(self.net, tgt + n2), tgt + n2)
            loss = (
                loss
                + compute_sisdr_loss(sh2, tgt, self.scale_w)
                + compute_sisdr_loss(nh2, n2, self.scale_w)
                + self.cycle_weight
                * compute_sisdr_loss(sh1, sh2.detach(), self.scale_w)
            )
        stats = {
            "gamma": torch.tensor(gamma),
            "generation": torch.tensor(float(self.generation)),
            "teacher_snr": self._bank(self.snr_bank).mean().detach(),
        }
        return loss, stats

    @torch.no_grad()
    def _diagnose_utmos(self, x):
        """UTMOS of the exported estimate and of the input (validation only)."""
        net = self.inference_net()
        est = read_speech(self._call(net, x), x, self.readout, self.tau_rel)
        if self._utmos is None:
            # kept out of the module tree: not trained, not in checkpoints
            self.__dict__["_utmos"] = _load_utmos()
        mos = self._utmos.to(x.device).eval()
        with torch.autocast(x.device.type, enabled=False):
            # UTMOS expects roughly unit-peak audio
            def score(w):
                w = w / w.abs().amax(-1, keepdim=True).clamp_min(1e-8) * 0.9
                return mos(w.float(), self.fs).float()

            m_est, m_in = score(est), score(x)
        return {"utmos": m_est.mean(), "utmos_delta": (m_est - m_in).mean()}

    @torch.no_grad()
    def _diagnose(self, x, s):
        """Oracle SI-SDR of what would be exported (validation only)."""
        net = self.inference_net()
        est = read_speech(self._call(net, x), x, self.readout, self.tau_rel)
        si = compute_si_sdr(est, s)
        return {"si_sdr": si.mean(), "si_sdri": (si - compute_si_sdr(x, s)).mean()}


def _load_utmos() -> nn.Module:
    """Load UTMOS22 strong (tarepan/SpeechMOS) from the torch hub cache."""
    hub = Path(torch.hub.get_dir()) / "tarepan_SpeechMOS_v1.2.0"
    if hub.is_dir():  # compute nodes may be offline: use the cached repo
        return torch.hub.load(str(hub), "utmos22_strong", source="local")
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


def _require(value, name: str, mode: str):
    if value is None:
        raise ValueError(f"mode {mode!r} needs `{name}` in every batch")
    return value


def _check_mode_slots(mode: str, net: EnhNet) -> None:
    need = {"dnf": (2,), "mixit": (3,), "supervised": (1, 2)}.get(mode)
    if need and net.num_outputs not in need:
        raise ValueError(
            f"mode {mode!r} needs a network with {need} output slot(s); this one "
            f"has {net.num_outputs} (set separator_conf.n_srcs / num_spk)"
        )


def _read_sidecar(init: str) -> Dict[str, Any]:
    from pathlib import Path

    import yaml

    path = Path(init) / "snoar.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
