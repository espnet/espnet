from espnet3.components.modeling.hf_models import AbsHFTrainingWrapper, AbsHFInferenceWrapper

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from typing import Tuple, Dict, List, Any


class GraniteSpeechModel(AbsHFTrainingWrapper):
    model_class = AutoModelForSpeechSeq2Seq

    def collect_feats(self, **batch) -> Dict[str, torch.Tensor]:
        feats = batch["input_features"]
        feats_lengths = batch["input_features_mask"].sum(dim=-1)
        return {"feats": feats, "feats_lengths": feats_lengths}


class GraniteSpeechInferenceSession(AbsHFInferenceWrapper):
    model_class = AutoModelForSpeechSeq2Seq

    def __init__(self, model_tag_or_path: str, user_prompt: str, **kwargs):
        super().__init__(model_tag_or_path, **kwargs)
        self.tokenizer = self.processor.tokenizer

        chat = [
            {"role": "user", "content": user_prompt},
        ]
        self.prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    def forward(self, speech):
        inputs = self.processor(text=self.prompt, audio=speech,
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
    def __init__(self, model_tag_or_path: str, user_prompt: str, **kwargs):
        self.processor = AutoProcessor.from_pretrained(model_tag_or_path)
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
