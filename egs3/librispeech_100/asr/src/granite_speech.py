from espnet2.torch_utils.device_funcs import force_gatherable

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import lightning
from typing import Tuple, Dict, List, Any


class GraniteSpeechModel(lightning.LightningModule):
    def __init__(self, model_tag: str, **kwargs):
        super().__init__()
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_tag)

    def forward(
        self,
        **batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        outputs = self.model(**batch)
        loss = outputs.loss
        stats = {"loss": loss}
        batch_size = batch["input_ids"].shape[0]

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(self, **batch):
        feats = batch["input_features"]
        feats_lengths = batch["input_features_mask"].sum(dim=-1)
        return {"feats": feats, "feats_lengths": feats_lengths}


class GraniteSpeechInferenceSession(lightning.LightningModule):
    def __init__(self, model_tag: str, user_prompt: str, checkpoint_path: str | None = None, **kwargs):
        super().__init__()
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint_path or model_tag)
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.tokenizer = self.processor.tokenizer

        chat = [
            {"role": "user", "content": user_prompt},
        ]
        self.prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    def forward(self, speech):
        inputs = self.processor(self.prompt, speech,
                                return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=200, do_sample=False, num_beams=1)

        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        return [output_text]


class GraniteSpeechCollateFn:
    def __init__(self, model_tag: str, user_prompt: str, **kwargs):
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.tokenizer = self.processor.tokenizer

        chat = [
            {"role": "user", "content": user_prompt},
        ]
        self.prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    def __call__(self, data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        batch_size = len(data)
        uids = [data[i]["utt_id"] for i in range(batch_size)]
        speech = [data[i]["speech"] for i in range(batch_size)]
        texts = [data[i]["text"] for i in range(batch_size)]

        inputs = self.processor(
            [self.prompt] * batch_size,
            audio=speech,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        # tokenize text
        targets = [t + self.processor.tokenizer.eos_token for t in texts]
        targets = self.tokenizer(targets, return_tensors="pt",
                                 padding=True, padding_side="right")

        # concatenate text with input for loss calculation
        input_ids = torch.cat([inputs.input_ids, targets.input_ids], dim=1)
        attention_mask = torch.cat(
            [inputs.attention_mask, targets.attention_mask], dim=1)

        labels = targets.input_ids.clone()
        labels[~(targets.attention_mask.bool())] = -100
        labels = torch.cat(
            [torch.full_like(inputs.input_ids, -100), labels], dim=1)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": inputs.input_features,
            "input_features_mask": inputs.input_features_mask
        }

        return uids, batch
