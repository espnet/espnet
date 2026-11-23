from threading import Thread
from typing import Any, Dict, Iterator, List, Optional

import torch
from typeguard import typechecked

from espnet2.sds.llm.abs_llm import AbsLLM


class HuggingFaceLLM(AbsLLM):
    """Hugging Face LLM"""

    @typechecked
    def __init__(
        self,
        access_token: Optional[str] = None,
        tag: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 64,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """A class for initializing a text response generator

        using the Transformers library.

        Args:
            access_token (str, optional):
                The Hugging Face access token. Only gated models such as
                meta-llama/* need it. Defaults to None.
            tag (str, optional):
                The model tag for the pre-trained language model.
                Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            device (str, optional):
                The device to run the inference on. Defaults to "cuda".
            dtype (str, optional):
                The data type for model computation, a torch dtype name or
                "auto" for the checkpoint's own. Defaults to "float16".
            max_new_tokens (int, optional):
                The maximum number of tokens in one reply. Defaults to 64.
            chat_template_kwargs (dict, optional):
                Extra variables for the model's chat template. Defaults to
                {"enable_thinking": False}, which stops Qwen3 and SmolLM3
                from writing their reasoning before the reply. A chat
                template that does not use a variable ignores it.

        Raises:
            ImportError:
                If the `transformers` library is not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            print(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )
            raise e
        super().__init__()
        self.device = device
        self.max_new_tokens = max_new_tokens
        if chat_template_kwargs is None:
            chat_template_kwargs = {"enable_thinking": False}
        self.chat_template_kwargs = chat_template_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(tag, token=access_token)
        self.model = (
            AutoModelForCausalLM.from_pretrained(tag, dtype=dtype, token=access_token)
            .to(device)
            .eval()
        )

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        dummy_chat = [
            {"role": "user", "content": "Write me a poem about Machine Learning."}
        ]
        self.forward(dummy_chat)

    def _encode(self, chat_messages: List[dict]) -> Dict[str, Any]:
        """Render the chat with the model's chat template and tokenize it.

        apply_chat_template tokenizes the rendered text itself. Tokenizing that
        text with the tokenizer instead would add the BOS token a second time on
        a model whose template already writes it, such as Llama 3.
        """
        return self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        ).to(self.device)

    def stream(self, chat_messages: List[dict]) -> Iterator[str]:
        """Yield the reply piece by piece while the model generates it.

        A caller that times the first non-empty piece gets the time to the
        first text of the reply, which forward() cannot report. A piece is
        decoded text, not one token: TextIteratorStreamer holds the text back
        until it ends a word, so the first piece can come a few tokens after
        the first token.

        Args:
            chat_messages (List[dict]):
                The conversation history, as for forward().

        Yields:
            str: The next decoded piece of the reply.
        """
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self._encode(chat_messages)
        error = []

        def generate():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        streamer=streamer,
                    )
            except Exception as e:
                # Record the error and end the stream. Without end(), the
                # loop below waits forever for a piece that never arrives.
                error.append(e)
                streamer.end()

        thread = Thread(target=generate)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()
        if error:
            raise error[0]

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
            - The model generates at most `max_new_tokens` new tokens
            with greedy decoding (`do_sample` set to False).
        """
        inputs = self._encode(chat_messages)
        with torch.no_grad():
            # Ask for a tensor: a checkpoint whose generation config sets
            # return_dict_in_generate would otherwise return a ModelOutput.
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=False,
            )
        n_prompt = inputs["input_ids"].size(1)
        return self.tokenizer.decode(
            output[0, n_prompt:], skip_special_tokens=True
        ).strip()
