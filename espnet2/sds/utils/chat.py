# Copyright 2024
# Licensed under the Apache-2.0 license
# For details, see: http://www.apache.org/licenses/

# This code is adapted from the following sources:
# 1. https://github.com/huggingface/speech-to-speech

from typing import Dict, List


class Chat:
    """Handles the chat to avoid OOM issues.

    Attributes:
        size (int):
            The maximum size of the buffer for chat pairs.
        init_chat_message (Union[Dict[str, str], None]):
            The initial message to prepend to the chat history,
            if any.
        buffer (List[Dict[str, str]]):
            A list of chat messages in the form of dictionaries,
            maintaining the chat history.
    """

    def __init__(self, size: int):
        """Initializes the Chat object with a buffer size.

        Args:
            size (int):
                The maximum number of chat pairs (prompt-response)
                to retain in the buffer.
        """
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we
        # add an prompt and assistant answer
        self.buffer = []

    def append(self, item: Dict[str, str]) -> None:
        """Adds a new message to the chat buffer.

        Removes the oldest prompt-response
        pair if the buffer exceeds the maximum allowed size.

        Args:
            item (Dict[str, str]):
                The chat message to be added to the buffer, typically
                with keys like "role" (e.g., "user", "assistant")
                and "content".
        """
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message: Dict[str, str]) -> None:
        """Sets the initial chat message, which will be prepended to the chat history.

        Args:
            init_chat_message (Dict[str, str]):
                The initial message to set, typically with keys
                like "role" and "content".
        """
        self.init_chat_message = init_chat_message

    def to_list(self) -> List[Dict[str, str]]:
        """Returns the complete chat history as a list,

        including the initial message if set.

        Returns:
            List[Dict[str, str]]:
                The chat history as a list of dictionaries,
                starting with the initial message
                if it is set, followed by the buffer contents.
        """
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer
