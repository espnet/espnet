from functools import lru_cache

import torch
import torch.nn as nn

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.legacy.nets.pytorch_backend.nets_utils import th_accuracy


def _maybe_apply_peft(model, peft):
    if peft is None:
        return model

    print(f"Applying PEFT: {peft}")
    if isinstance(peft, dict):
        peft_type = peft.get("type")
        if peft_type in ("espnet_lora", "espnet2_lora", "lora_espnet"):
            try:
                from espnet2.layers.create_adapter_fn import create_lora_adapter
            except Exception as exc:
                raise ImportError(
                    "ESPnet LoRA is requested but espnet2.layers.create_adapter_fn is not available."
                ) from exc
            peft = dict(peft)
            peft.pop("type", None)
            return create_lora_adapter(model, **peft)

    try:
        from peft import (
            AdaLoraConfig,
            DeloraConfig,
            LoraConfig,
            PeftModel,
            RandLoraConfig,
            TaskType,
            VBLoRAConfig,
            XLoraConfig,
            get_peft_model,
        )
    except Exception as exc:
        raise ImportError(
            "PEFT is requested but the 'peft' package is not available. "
            "Install peft or set peft=None."
        ) from exc

    if isinstance(peft, str):
        return PeftModel.from_pretrained(model, peft)

    if isinstance(peft, dict):
        peft = dict(peft)
        if "pretrained" in peft:
            adapter_path = peft.pop("pretrained")
            return PeftModel.from_pretrained(model, adapter_path, **peft)

        peft_type = peft.pop("type", "lora")

        config_cls_map = {
            "lora": LoraConfig,
            "adalora": AdaLoraConfig,
            "delora": DeloraConfig,
            "randlora": RandLoraConfig,
            "vblora": VBLoRAConfig,
            "xlora": XLoraConfig,
        }
        config_cls = config_cls_map.get(peft_type)
        if config_cls is None:
            raise ValueError(f"Unsupported PEFT type: {peft_type}")

        task_type = peft.pop("task_type", None)
        if peft_type in ("lora", "adalora"):
            if isinstance(task_type, str):
                task_type = getattr(TaskType, task_type.upper())
            if task_type is None:
                task_type = (
                    TaskType.SEQ_2_SEQ_LM
                    if hasattr(model, "generate")
                    else TaskType.FEATURE_EXTRACTION
                )
            config = config_cls(task_type=task_type, **peft)
        else:
            # Other PEFT configs do not accept task_type.
            config = config_cls(**peft)

        # For non-transformers models (e.g., ESPnet), avoid PeftModel wrappers
        # that expect generation helpers like prepare_inputs_for_generation.
        if not hasattr(model, "prepare_inputs_for_generation") and not hasattr(
            model, "generate"
        ):
            from peft.tuners import (
                AdaLoraModel,
                DeloraModel,
                LoraModel,
                RandLoraModel,
                VBLoRAModel,
                XLoraModel,
            )

            tuner_cls_map = {
                "lora": LoraModel,
                "adalora": AdaLoraModel,
                "delora": DeloraModel,
                "randlora": RandLoraModel,
                "vblora": VBLoRAModel,
                "xlora": XLoraModel,
            }
            tuner_cls = tuner_cls_map[peft_type]
            return tuner_cls(model, config, "default")
        return get_peft_model(model, config)

    return get_peft_model(model, peft)


import gc


class OWSMFinetune(nn.Module):
    def __init__(self, model_tag, peft=None):
        super().__init__()
        owsm_model = Speech2Text.from_pretrained(model_tag)
        m = _maybe_apply_peft(owsm_model.s2t_model, peft)

        total_params = sum(p.numel() for p in owsm_model.s2t_model.parameters())
        trainable_params = sum(
            p.numel() for p in owsm_model.s2t_model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        if m is not None:
            self.model = m
        else:
            self.model = owsm_model.s2t_model

    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        text_ctc,
        text_ctc_lengths,
        text_prev,
        text_prev_lengths,
    ):
        return self.model(
            speech,
            speech_lengths,
            text,
            text_lengths,
            text_prev,
            text_prev_lengths,
            text_ctc,
            text_ctc_lengths,
        )

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        if self.model is not None:
            del self.model
            gc.collect()
            self.model = None
        return {"feats": speech, "feats_lengths": speech_lengths}


class WhisperFinetune(nn.Module):
    def __init__(self, model_tag, peft=None):
        super().__init__()
        # get whisper model and preprocessor from transformers
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_tag)
        self.model = _maybe_apply_peft(self.model, peft)
        self.model = self.model.to(
            torch.float32
        )  # use float32 for stability, can be changed to bf16 later

        # init error calculator
        from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator

        # get token_list from whisper model
        token_list = self.processor.tokenizer.get_vocab()
        token_list = sorted(token_list, key=token_list.get)
        # we will not use them. init by random
        sym_space, sym_blank = "<space>", "<blank>"
        self.error_calculator = ErrorCalculator(
            char_list=token_list,
            sym_space=sym_space,
            sym_blank=sym_blank,
            report_cer=True,
            report_wer=True,
        )

    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        **kwargs,
    ):
        # add here: make sure speech_lengths is tensor on correct device + clamp
        if not torch.is_tensor(speech_lengths):
            speech_lengths = torch.as_tensor(speech_lengths, device=speech.device)
        speech_lengths = speech_lengths.to(device=speech.device, dtype=torch.long)
        speech_lengths = torch.clamp(speech_lengths, max=3000)

        # transpose back to (B, D, T') for whisper
        speech = speech.transpose(1, 2)  # (B, D, T')
        # pad to 30 seconds (3000 frames after processing)
        speech = torch.nn.functional.pad(
            speech, (0, max(0, 3000 - speech.size(2))), value=0.0
        )[
            :, :, :3000
        ]  # (B, D, 3000)
        attention_mask = torch.arange(3000).expand(len(speech_lengths), 3000).to(
            speech.device
        ) < speech_lengths.unsqueeze(
            1
        )  # (B, 3000)

        # make decoder input ids and labels
        decoder_input_ids = text[:, :-1][
            :, : self.model.config.max_target_positions
        ]  # (B, L-1)
        labels = text[:, 1:][:, : self.model.config.max_target_positions]  # (B, L-1)
        labels = labels.clone()  # add dahee
        labels[labels < 0] = -100  # add dahee

        output = self.model(
            input_features=speech,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        # breakpoint()
        loss = output.loss
        # acc = th_accuracy(output.logits.reshape(-1, output.logits.size(-1)), labels, ignore_label=50256)        # 50256 is ""
        acc = th_accuracy(
            output.logits.reshape(-1, output.logits.size(-1)), labels, ignore_label=-100
        )  # 50256 is ""
        cer_att, wer_att = None, None
        if not self.training:
            ys_hat = output.logits.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(
                ys_hat.detach().cpu().numpy(), labels.detach().cpu().numpy()
            )
            cer_att, wer_att = torch.tensor(cer_att), torch.tensor(wer_att)
        stats = {
            "loss": loss,
            "acc": torch.tensor(acc),
            "cer_att": cer_att,
            "wer_att": wer_att,
        }
        return loss, stats, torch.tensor(speech.size(0))

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}


class OWSMV4BaseInferenceModel(nn.Module):
    def __init__(
        self,
        *,
        model_tag: str,
        lang_sym: str,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.s2t = Speech2Text.from_pretrained(
            model_tag=model_tag,
            lang_sym=lang_sym,
            device=str(device),
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            # breakpoint()
            self.s2t.s2t_model.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in state.items()
                    if k.startswith("model.")
                }
            )

    def forward(self, speech):
        return {"text": self.s2t(speech)}


class WhisperInferenceModel(nn.Module):
    def __init__(self, model_tag, peft=None, checkpoint_path=None, device="cuda"):
        super().__init__()
        from transformers import (
            AutoProcessor,
            GenerationConfig,
            WhisperConfig,
            WhisperForConditionalGeneration,
        )

        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.config = WhisperConfig.from_pretrained(model_tag)
        self.model = WhisperForConditionalGeneration(self.config)
        self.model = _maybe_apply_peft(self.model, peft)
        self.model.generation_config = GenerationConfig.from_pretrained(model_tag)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
            self.load_state_dict(state, strict=False)

        self.model = self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

    def forward(self, speech):
        """
        speech: Tensor of shape (1, T) or (T,)
        """
        # speech = speech.astype(torch.float32)
        processed = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30 * 16000,
        )

        input_features = processed["input_features"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features=input_features,
                num_beams=1,
                language="pt",
                task="transcribe",
                max_new_tokens=128,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return {"text": text[0]}
