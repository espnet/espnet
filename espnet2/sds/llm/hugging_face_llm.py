from espnet2.sds.llm.abs_llm import AbsLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import torch
from typeguard import typechecked

class HuggingFaceLLM(AbsLLM):
    """Hugging Face LLM"""

    @typechecked
    def __init__(
        self,
        access_token,
        tag = "meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
    ): 
        super().__init__()
        LM_tokenizer = AutoTokenizer.from_pretrained(tag, token=access_token)
        LM_model = AutoModelForCausalLM.from_pretrained(
            tag, torch_dtype="float16", trust_remote_code=True, token=access_token
        ).to(device)
        self.LM_pipe = pipeline(
            "text-generation", model=LM_model, tokenizer=LM_tokenizer, device=device
        )
    
    def warmup(self):
        with torch.no_grad():
            dummy_input_text = "Write me a poem about Machine Learning."
            dummy_chat = [{"role": "user", "content": dummy_input_text}]
            self.LM_pipe(
                dummy_chat,
                max_new_tokens=32,
                min_new_tokens=0,
                temperature=0.0,
                do_sample=False,
            )
    
    def forward(self,chat_messages):
        with torch.no_grad():
            output=self.LM_pipe(
                chat_messages,
                max_new_tokens=64,
                min_new_tokens=0,
                temperature=0.0,
                do_sample=False,
            )
            generated_text = output[0]['generated_text'][-1]["content"]
            return generated_text

