from abc import ABC, abstractmethod

import torch


class AbsPostDecoder(torch.nn.Module, ABC):
    """
    Abstract base class for post-decoders in speech language understanding (SLU).

    This class serves as a blueprint for creating post-decoder modules that
    process input transcripts and produce output tensors. It defines the
    necessary methods that must be implemented by any derived class.

    Attributes:
        None

    Methods:
        output_size() -> int:
            Returns the size of the output tensor produced by the decoder.
        
        forward(transcript_input_ids: torch.LongTensor,
                transcript_attention_mask: torch.LongTensor,
                transcript_token_type_ids: torch.LongTensor,
                transcript_position_ids: torch.LongTensor) -> torch.Tensor:
            Processes the input transcript data and produces the output tensor.
        
        convert_examples_to_features(data: list, max_seq_length: int,
                                      output_size: int):
            Converts a list of data examples into model input features.

    Raises:
        NotImplementedError:
            If the derived class does not implement the abstract methods.

    Examples:
        class CustomPostDecoder(AbsPostDecoder):
            def output_size(self) -> int:
                return 10

            def forward(self, transcript_input_ids, transcript_attention_mask,
                        transcript_token_type_ids, transcript_position_ids):
                # Implement forward logic here
                return torch.randn(1, self.output_size())

            def convert_examples_to_features(self, data, max_seq_length, output_size):
                # Implement conversion logic here
                return processed_data

        decoder = CustomPostDecoder()
        print(decoder.output_size())  # Output: 10
    """
    @abstractmethod
    def output_size(self) -> int:
        """
        Returns the size of the output tensor.

        This method must be implemented by subclasses of the AbsPostDecoder class. 
        The output size typically corresponds to the number of classes or the 
        dimensionality of the output space for the specific decoding task.

        Returns:
            int: The size of the output tensor.

        Examples:
            # Example subclass implementation
            class MyPostDecoder(AbsPostDecoder):
                def output_size(self) -> int:
                    return 10  # Assuming the output size is 10 classes

            decoder = MyPostDecoder()
            print(decoder.output_size())  # Output: 10

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        transcript_input_ids: torch.LongTensor,
        transcript_attention_mask: torch.LongTensor,
        transcript_token_type_ids: torch.LongTensor,
        transcript_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Processes the input tensors through the model to produce output.

        This method takes various input tensors related to the transcript and 
        applies the necessary transformations to produce a final output tensor.

        Args:
            transcript_input_ids (torch.LongTensor): The input IDs for the 
                transcript, typically the tokenized representation of the text.
            transcript_attention_mask (torch.LongTensor): A mask to indicate 
                which tokens should be attended to, with 1s for tokens that 
                should be processed and 0s for padding tokens.
            transcript_token_type_ids (torch.LongTensor): A tensor that 
                distinguishes between different segments in the input (e.g., 
                for tasks that involve multiple sentences).
            transcript_position_ids (torch.LongTensor): A tensor that provides 
                the position of each token in the sequence, used for positional 
                encoding.

        Returns:
            torch.Tensor: The output tensor after processing the input through 
            the model, typically containing logits or class probabilities.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> decoder = MyPostDecoder()  # MyPostDecoder is a concrete subclass
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> attention_mask = torch.tensor([[1, 1, 1]])
            >>> token_type_ids = torch.tensor([[0, 0, 0]])
            >>> position_ids = torch.tensor([[0, 1, 2]])
            >>> output = decoder.forward(input_ids, attention_mask, 
            ...                          token_type_ids, position_ids)
            >>> print(output.shape)  # Expected output shape depends on the model

        Note:
            This method must be overridden in any subclass of 
            `AbsPostDecoder`. Ensure that the specific implementation 
            adheres to the expected input and output formats.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_examples_to_features(
        self, data: list, max_seq_length: int, output_size: int
    ):
        """
        Converts a list of examples into model input features.

        This method takes raw data examples and processes them into features 
        suitable for model input. The conversion includes tokenization, 
        padding, and truncation based on the specified maximum sequence length.

        Args:
            data (list): A list of examples, where each example is a 
                dictionary containing input data necessary for the model.
            max_seq_length (int): The maximum length of the input sequences. 
                Sequences longer than this will be truncated, and shorter 
                sequences will be padded.
            output_size (int): The size of the output that the model will 
                generate. This is typically determined by the model's 
                architecture.

        Returns:
            list: A list of features, where each feature is a dictionary 
                containing the processed input tensors (input_ids, 
                attention_mask, etc.) ready for model consumption.

        Raises:
            ValueError: If the input data is not formatted correctly or 
                if any of the inputs are invalid.

        Examples:
            >>> decoder = MyPostDecoder()  # Assume MyPostDecoder implements AbsPostDecoder
            >>> examples = [{"text": "Hello world!"}, {"text": "How are you?"}]
            >>> features = decoder.convert_examples_to_features(examples, 
            ...                                                  max_seq_length=10, 
            ...                                                  output_size=5)
            >>> print(features)
            [{'input_ids': [...], 'attention_mask': [...], ...}, ...]

        Note:
            The specific implementation of this method should handle the 
            actual logic for converting the raw data examples to features 
            based on the model's requirements.

        Todo:
            Implement validation checks for the input data format.
        """
        raise NotImplementedError
