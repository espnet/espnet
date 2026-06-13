"""GAN-TTS adapter for ESPnet3 Lightning integration."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from espnet2.gan_tts.espnet_model import ESPnetGANTTSModel
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep


def _patch_gan_tts_collect_feats() -> None:
    """
    espnet3's collect_stats only compute shape for keys returned by
    ``collect_feats``. By default GAN-TTS only returns ``feats``, so
    ``speech_shape`` / ``text_shape`` are missing. These shapes are
    needed for dataloader sampling.
    """
    # This prevents double-patching if collect_stats is called multiple times.
    if getattr(ESPnetGANTTSModel.collect_feats, "_patched_for_input_shapes", False):
        return

    original = ESPnetGANTTSModel.collect_feats

    def collect_feats_with_inputs(self, **kwargs):
        out = original(self, **kwargs)
        if "speech" in kwargs and "speech_lengths" in kwargs:
            out["speech"] = kwargs["speech"]
            out["speech_lengths"] = kwargs["speech_lengths"]
        if "text" in kwargs and "text_lengths" in kwargs:
            out["text"] = kwargs["text"]
            out["text_lengths"] = kwargs["text_lengths"]
        return out

    collect_feats_with_inputs._patched_for_input_shapes = True
    ESPnetGANTTSModel.collect_feats = collect_feats_with_inputs


class GANTTSLightningModule(ESPnetLightningModule):
    """Adapt ESPnet2 GAN-TTS model outputs to ESPnet3's named optimizer path."""

    _GAN_STEP_NAMES = {
        0: "generator",
        1: "discriminator",
    }

    def __init__(self, model, config):
        super().__init__(model, config)
        self.automatic_optimization = False

    def collect_stats(self):
        """Apply the GAN-TTS collect_feats patch.

        This patch make the collect_stats method also return
        the input speech/text features, which are needed for dataloader sampling.
        """
        _patch_gan_tts_collect_feats()
        return super().collect_stats()

    def _gan_option(self, name: str, default):
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None and hasattr(trainer_cfg, "gan"):
            gan_cfg = trainer_cfg.gan
            if hasattr(gan_cfg, name):
                return getattr(gan_cfg, name)
            if isinstance(gan_cfg, dict):
                return gan_cfg.get(name, default)
        return default

    def _turns_in_order(self) -> List[tuple[str, bool]]:
        generator_first = bool(self._gan_option("generator_first", False))
        if generator_first:
            return [("generator", True), ("discriminator", False)]
        return [("discriminator", False), ("generator", True)]

    def _normalize_optim_idx(self, optim_idx) -> int:
        if isinstance(optim_idx, int):
            normalized = optim_idx
        elif isinstance(optim_idx, torch.Tensor):
            if optim_idx.dim() >= 2:
                raise AssertionError(
                    "GAN-TTS optim_idx must be an int or a 0/1-dim tensor, "
                    f"but got {optim_idx.dim()} dims."
                )
            if optim_idx.dim() == 1:
                if optim_idx.numel() == 0:
                    raise AssertionError("GAN-TTS optim_idx tensor must not be empty.")
                first = optim_idx[0]
                if not torch.all(optim_idx == first):
                    raise AssertionError(
                        "GAN-TTS optim_idx 1-dim tensor must have identical values."
                    )
                normalized = int(first.item())
            else:
                normalized = int(optim_idx.item())
        else:
            raise AssertionError(
                "GAN-TTS optim_idx must be an int or torch.Tensor, "
                f"but got {type(optim_idx)!r}."
            )

        if normalized not in self._GAN_STEP_NAMES:
            raise AssertionError(
                "GAN-TTS models must set optim_idx to 0 (generator) or 1 "
                f"(discriminator), but got {normalized!r}."
            )
        return normalized

    def _clear_model_cache(self) -> None:
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

    def _should_skip_discriminator(self, mode: str) -> bool:
        if mode != "train":
            return False

        skip_prob = float(self._gan_option("skip_discriminator_prob", 0.0) or 0.0)
        if skip_prob <= 0.0:
            return False

        skip_disc = torch.rand(1, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(skip_disc, src=0)
        return bool(skip_disc.item() < skip_prob)

    def _forward_gan_turn(
        self, batch, forward_generator: bool
    ) -> Tuple[OptimizationStep, Dict[str, torch.Tensor], torch.Tensor | None]:
        output = self.model(**batch[1], forward_generator=forward_generator)
        if not isinstance(output, dict):
            raise AssertionError(
                "GAN-TTS models must return a dict with loss/stats/weight/optim_idx."
            )

        optim_idx = self._normalize_optim_idx(output.get("optim_idx"))
        name = self._GAN_STEP_NAMES[optim_idx]

        loss = output.get("loss")
        if not isinstance(loss, torch.Tensor):
            raise AssertionError(
                f"GAN-TTS '{name}' loss must be a tensor, but got {type(loss)!r}."
            )

        stats = output.get("stats")
        if not isinstance(stats, dict):
            raise AssertionError(
                f"GAN-TTS '{name}' stats must be a dict, but got {type(stats)!r}."
            )

        weight = output.get("weight")
        return OptimizationStep(loss=loss, name=name), stats, weight

    def _run_gan_optimizer_update(
        self,
        step: OptimizationStep,
        stats: Dict[str, torch.Tensor],
        weight,
        batch_idx: int,
        turn_name: str,
        forward_time: float,
    ) -> None:
        named_optimizers = self._get_named_optimizers()
        spec_by_name = {spec.name: spec for spec in self._optimizer_specs}
        name_to_idx = {
            name: idx for idx, name in enumerate(self._multi_optimizer_names)
        }

        if step.name not in spec_by_name:
            raise AssertionError(
                f"Unknown optimizer '{step.name}'. Available optimizers: "
                f"{', '.join(sorted(spec_by_name))}."
            )

        spec = spec_by_name[step.name]
        state = self._optimizer_states[step.name]
        optimizer = named_optimizers[step.name]
        optim_idx = name_to_idx[step.name]

        extra_stats: Dict[str, torch.Tensor | float] = {
            f"{step.name}/loss": step.loss.detach(),
            f"{turn_name}_forward_time": forward_time,
        }

        if state.accum_counter == 0:
            optimizer.zero_grad()

        backward_start = time.perf_counter()
        self.manual_backward(step.loss / spec.accum_grad_steps)
        extra_stats[f"{turn_name}_backward_time"] = time.perf_counter() - backward_start
        state.accum_counter += 1

        meets_accum = state.accum_counter >= spec.accum_grad_steps
        meets_iter = (batch_idx + 1) % spec.step_every_n_iters == 0

        optim_step_time = 0.0
        if meets_accum and meets_iter:
            if spec.gradient_clip_val is not None:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=spec.gradient_clip_val,
                    gradient_clip_algorithm=spec.gradient_clip_algorithm,
                )

            optim_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            optim_step_time = time.perf_counter() - optim_start
            state.accum_counter = 0
            state.update_step += 1
            extra_stats[f"{step.name}/update_step"] = float(state.update_step)
            self._step_named_scheduler_on_update(step.name)

            for group_idx, group in enumerate(optimizer.param_groups):
                if "lr" in group:
                    extra_stats[f"optim{optim_idx}_lr{group_idx}"] = float(group["lr"])

        extra_stats[f"{turn_name}_optim_step_time"] = optim_step_time
        extra_stats[f"{turn_name}_train_time"] = (
            forward_time + extra_stats[f"{turn_name}_backward_time"] + optim_step_time
        )

        self._log_stats("train", stats, weight, extra_stats=extra_stats)

    def _step(self, batch, batch_idx, mode):
        if not isinstance(self.model, AbsGANESPnetModel):
            return super()._step(batch, batch_idx, mode)

        if bool(self._gan_option("no_forward_run", False)):
            return None

        for turn_name, forward_generator in self._turns_in_order():
            if turn_name == "discriminator" and self._should_skip_discriminator(mode):
                self._clear_model_cache()
                continue

            forward_start = time.perf_counter()
            step, stats, weight = self._forward_gan_turn(batch, forward_generator)
            forward_time = time.perf_counter() - forward_start

            any_invalid = self._check_nan_inf_loss([step.loss], batch_idx)
            if any_invalid:
                return None

            if mode == "train":
                self._run_gan_optimizer_update(
                    step=step,
                    stats=stats,
                    weight=weight,
                    batch_idx=batch_idx,
                    turn_name=turn_name,
                    forward_time=forward_time,
                )
            else:
                extra_stats = {
                    f"{step.name}/loss": step.loss.detach(),
                    f"{turn_name}_forward_time": forward_time,
                }
                self._log_stats(mode, stats, weight, extra_stats=extra_stats)
        return None
