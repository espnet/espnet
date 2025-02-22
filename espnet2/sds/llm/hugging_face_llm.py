from typing import List

import torch
from typeguard import typechecked

from espnet2.sds.llm.abs_llm import AbsLLM


class HuggingFaceLLM(AbsLLM):
    """Hugging Face LLM"""

    @typechecked
    def __init__(
        self,
        access_token: str,
        tag: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """A class for initializing a text response generator

        using the Transformers library.

        Args:
            access_token (str):
                The access token required for downloading models from Hugging Face.
            tag (str, optional):
                The model tag for the pre-trained language model.
                Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            device (str, optional):
                The device to run the inference on. Defaults to "cuda".
            dtype (str, optional):
                The data type for model computation. Defaults to "float16".

        Raises:
            ImportError:
                If the `transformers` library is not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as e:
            print(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )
            raise e
        super().__init__()
        LM_tokenizer = AutoTokenizer.from_pretrained(tag, token=access_token)
        LM_model = AutoModelForCausalLM.from_pretrained(
            tag, torch_dtype=dtype, trust_remote_code=True, token=access_token
        ).to(device)
        self.LM_pipe = pipeline(
            "text-generation", model=LM_model, tokenizer=LM_tokenizer, device=device
        )

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
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

    def forward(self, chat_messages: List[dict]) -> str:
        """Generate a response from the language model based on

        the provided chat messages.

        Args:
            chat_messages (List[dict]):
                A list of chat messages, where each message is a
                dictionary containing the
                conversation history. Each dictionary should have
                keys like "role" (e.g., "user", "assistant")
                and "content" (the message text).

        Returns:
            str:
                The generated response text from the language model.

        Notes:
            - The model generates a response with a maximum of 64
            new tokens and a deterministic sampling strategy
            (temperature set to 0 and `do_sample` set to False).
        """
        with torch.no_grad():
            output = self.LM_pipe(
                chat_messages,
                max_new_tokens=64,
                min_new_tokens=0,
                temperature=0.0,
                do_sample=False,
            )
            generated_text = output[0]["generated_text"][-1]["content"]
            return generated_text
