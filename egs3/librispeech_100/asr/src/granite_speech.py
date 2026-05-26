from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.asr.espnet_model import ESPnetASRModel

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
import torch
from typing import Tuple, Dict, List, Any


class GraniteSpeechModel(ESPnetASRModel):
    def __init__(self, model_tag: str, user_prompt: str):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_tag)

    def forward(
        self,
        batch: BatchFeature,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        outputs = self.model(**batch)
        loss = outputs.loss
        stats = {"loss": loss}
        batch_size = batch["input_ids"].shape[0]

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


class GraniteSpeechCollateFn:
    def __init__(self, model_tag: str, user_prompt: str):
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.tokenizer = self.processor.tokenizer

        chat = [
            {"role": "user", "content": user_prompt},
        ]
        self.prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    def __call__(self, data: List[Tuple[str, Dict[str, Any]]]) -> BatchFeature:
        n = len(data)
        uids = [data[i][0] for i in range(n)]
        speech = [data[i][1]["speech"] for i in range(n)]
        texts = [data[i][1]["text"] for i in range(n)]

        inputs = self.processor(
            [self.prompt] * n,
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
        labels = torch.cat([torch.full_like(inputs.input_ids, -100), labels], dim=1)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": inputs.input_features,
            "input_features_mask": inputs.input_features_mask
        }

        return uids, batch
