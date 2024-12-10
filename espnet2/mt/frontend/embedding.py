#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Embedding(AbsFrontend):
    """
        Embedding Frontend for text based inputs.

    This class provides an embedding layer for processing text inputs,
    utilizing positional encoding to enhance the representation of input
    tokens.

    Attributes:
        embed_dim (int): The dimension of the embedding space.
        embed (torch.nn.Sequential): A sequential model combining embedding and
            positional encoding.

    Args:
        input_size (int): Number of input tokens.
        embed_dim (int): Embedding size.
        pos_enc_class: Class for positional encoding (e.g., PositionalEncoding).
        positional_dropout_rate (float): Dropout rate after adding positional
            encoding.

    Returns:
        None

    Examples:
        >>> embedding = Embedding(input_size=1000, embed_dim=256)
        >>> input_tensor = torch.randint(0, 1000, (32, 50))  # (batch_size, seq_len)
        >>> input_lengths = torch.full((32,), 50)  # All sequences are of length 50
        >>> output, output_lengths = embedding(input_tensor, input_lengths)
        >>> output.shape  # Should be (32, 50, 256)
        torch.Size([32, 50, 256])
        >>> output_lengths.shape  # Should be (32,)
        torch.Size([32])
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        # TODO(sdalmia): check for padding idx
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, embed_dim),
            pos_enc_class(embed_dim, positional_dropout_rate),
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Apply a sliding window on the input.

        This method processes the input tensor and applies an embedding layer
        followed by positional encoding, returning the embedded output along
        with the input lengths.

        Args:
            input: A tensor of shape (B, T) or (B, T, D), where B is the batch size,
                T is the sequence length, and D is the feature dimension.
            input_lengths: A tensor containing the lengths of the input sequences
                within the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor with dimensions (B, T, D) representing the embedded output.
                - A tensor containing the output lengths within the batch.

        Examples:
            >>> embedding = Embedding(input_size=1000, embed_dim=256)
            >>> input_tensor = torch.randint(0, 1000, (32, 10))  # Batch of 32, seq len 10
            >>> input_lengths = torch.full((32,), 10)  # All sequences have length 10
            >>> output, output_lengths = embedding(input_tensor, input_lengths)
            >>> output.shape
            torch.Size([32, 10, 256])  # Output shape should match (B, T, D)
            >>> output_lengths
            tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

        Note:
            Ensure that the input tensor contains valid token indices within the
            range of the input size.
        """
        x = self.embed(input)

        return x, input_lengths

    def output_size(self) -> int:
        """
        Return output length of feature dimension D, i.e. the embedding dim.

        This method provides the size of the output feature dimension D, which is
        equivalent to the embedding dimension of the layer. It is useful for
        determining the output shape of the embedding layer, particularly when
        constructing models that depend on the embedding size.

        Returns:
            int: The embedding dimension size.

        Examples:
            >>> embedding = Embedding(input_size=500, embed_dim=256)
            >>> embedding.output_size()
            256

            >>> patch_embedding = PatchEmbedding(input_size=500, embed_dim=128)
            >>> patch_embedding.output_size()
            128

            >>> codec_embedding = CodecEmbedding(input_size=500)
            >>> codec_embedding.output_size()
            <codebook_dim value>  # replace with actual codebook dimension
        """
        return self.embed_dim


class PatchEmbedding(AbsFrontend):
    """
    Embedding Frontend for text based inputs.

    This class implements an embedding layer that processes input tokens in
    patches, allowing for a specified number of tokens per frame. It utilizes
    a specified positional encoding class and applies layer normalization
    after embedding the input.

    Attributes:
        embed_dim (int): Dimension of the embedding.
        token_per_frame (int): Number of tokens per frame in the input.
        emb (torch.nn.Embedding): The embedding layer.
        pos (PositionalEncoding): The positional encoding layer.
        ln (torch.nn.LayerNorm): Layer normalization layer.

    Args:
        input_size (int): Number of input tokens. Defaults to 400.
        embed_dim (int): Embedding size. Defaults to 400.
        token_per_frame (int): Number of tokens per frame in the input.
            Defaults to 1.
        pos_enc_class: Class for positional encoding, either
            PositionalEncoding or ScaledPositionalEncoding. Defaults to
            PositionalEncoding.
        positional_dropout_rate (float): Dropout rate after adding
            positional encoding. Defaults to 0.1.

    Raises:
        AssertionError: If input dimensions or lengths are invalid.

    Examples:
        >>> patch_embedding = PatchEmbedding(input_size=500, embed_dim=256)
        >>> input_tensor = torch.randint(0, 500, (32, 16))  # Batch of 32
        >>> input_lengths = torch.full((32,), 16)  # All sequences of length 16
        >>> output, output_lengths = patch_embedding(input_tensor, input_lengths)
        >>> output.shape  # Should be (32, 16 // token_per_frame, 256)
        torch.Size([32, 16, 256])
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        token_per_frame: int = 1,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            token_per_frame: number of tokens per frame in the input
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.token_per_frame = token_per_frame

        self.emb = torch.nn.Embedding(input_size, embed_dim)
        self.pos = pos_enc_class(embed_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(embed_dim)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Embedding Frontend for text based inputs.

        This class is designed to perform patch embedding on the input sequences.
        It applies a sliding window mechanism to the input and uses an embedding
        layer followed by positional encoding and layer normalization.

        Attributes:
            embed_dim (int): The dimensionality of the embedding space.
            token_per_frame (int): The number of tokens per frame in the input.

        Args:
            input_size (int): Number of input tokens. Default is 400.
            embed_dim (int): Embedding size. Default is 400.
            token_per_frame (int): Number of tokens per frame in the input. Default is 1.
            pos_enc_class: Class for positional encoding (default: PositionalEncoding).
            positional_dropout_rate (float): Dropout rate after adding positional encoding.
                Default is 0.1.

        Raises:
            AssertionError: If the input tensor's dimensions or lengths are invalid.

        Examples:
            >>> import torch
            >>> model = PatchEmbedding(input_size=500, embed_dim=256, token_per_frame=4)
            >>> input_tensor = torch.randint(0, 500, (8, 16))  # (B, T)
            >>> input_lengths = torch.tensor([16] * 8)  # Lengths for each batch
            >>> output, output_lengths = model(input_tensor, input_lengths)
            >>> print(output.shape)  # Output shape should be (8, 4, 256)
            >>> print(output_lengths)  # Output lengths should be (8,)

        Note:
            Ensure that the input tensor's second dimension is divisible by
            `token_per_frame`, and that input lengths are also valid.
        """

        assert input.dim() == 2, input.size()
        assert input.size(1) % self.token_per_frame == 0, input.size()
        assert torch.all(input_lengths % self.token_per_frame == 0), input_lengths

        B, T = input.size()
        x = input.view(B, T // self.token_per_frame, self.token_per_frame)
        x = self.emb(x).mean(dim=2)
        x = self.ln(self.pos(x))

        input_lengths = input_lengths // self.token_per_frame

        return x, input_lengths

    def output_size(self) -> int:
        """
        Return output length of feature dimension D, i.e. the embedding dim.

        This method provides the dimensionality of the output feature vector
        produced by the embedding layer. The output size is equal to the
        embedding dimension defined during the initialization of the
        PatchEmbedding class.

        Returns:
            int: The size of the output feature dimension, which is equal to
            the embedding dimension (embed_dim).

        Examples:
            >>> patch_embedding = PatchEmbedding(embed_dim=512)
            >>> output_dim = patch_embedding.output_size()
            >>> print(output_dim)
            512
        """
        return self.embed_dim


class CodecEmbedding(AbsFrontend):
    """
    Use codec dequantization process and the input embeddings.

    This class implements a codec embedding layer that utilizes a
    pre-trained codec model for audio processing. It applies a
    dequantization process to the input embeddings, allowing for
    effective feature extraction from quantized audio data.

    Attributes:
        quantizer: The quantizer from the pre-trained codec model.
        codebook_size: The size of the codebook used in the codec.
        codebook_dim: The dimensionality of the codebook.
        token_bias: The index of the first codec code.
        token_per_frame: The number of tokens per frame in the input.
        vocab_size: The size of the input vocabulary.
        pos: Positional encoding layer.
        ln: Layer normalization layer.
        decoder: Decoder from the pre-trained codec model.

    Args:
        input_size: Size of the input vocabulary.
        hf_model_tag: HuggingFace model tag for Espnet codec models.
        token_bias: The index of the first codec code.
        token_per_frame: Number of tokens per frame in the input.
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding class.
        positional_dropout_rate: Dropout rate after adding positional encoding.

    Raises:
        AssertionError: If input dimensions or lengths are invalid.

    Examples:
        >>> codec_embedding = CodecEmbedding(input_size=512)
        >>> input_tensor = torch.randint(0, 512, (8, 64))  # (batch_size, seq_len)
        >>> input_lengths = torch.full((8,), 64)  # All sequences are of length 64
        >>> output, lengths = codec_embedding(input_tensor, input_lengths)

    Note:
        The `input` tensor must have dimensions of (B, T) where B is the batch
        size and T is the total number of tokens. Additionally, the length of
        input tensors must be divisible by `token_per_frame`.
    """

    @typechecked
    def __init__(
        self,
        input_size,
        hf_model_tag: str = "espnet/amuse_encodec_16k",
        token_bias: int = 2,
        token_per_frame: int = 8,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            hf_model_tag: HuggingFace model tag for Espnet codec models
            token_bias: the index of the first codec code
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()

        from espnet2.bin.gan_codec_inference import AudioCoding

        model = AudioCoding.from_pretrained(model_tag=hf_model_tag).model
        self.quantizer = model.codec.generator.quantizer
        self.codebook_size = self.quantizer.bins
        self.codebook_dim = self.quantizer.codebook_dim
        self.token_bias = token_bias

        # NOTE(Jinchuan): make it as an external parameter rather than parsing from
        # the quantizer since not all codebooks will be used all the time.
        self.token_per_frame = token_per_frame

        self.vocab_size = input_size
        self.pos = pos_enc_class(self.codebook_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(self.codebook_dim)

        self.decoder = model.codec.generator.decoder

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        """
            Use codec dequantization process and the input embeddings.

        This class implements an embedding frontend that utilizes a codec
        dequantization process to transform input embeddings. It incorporates
        positional encoding and layer normalization to process the input data.

        Attributes:
            hf_model_tag (str): HuggingFace model tag for Espnet codec models.
            token_bias (int): The index of the first codec code.
            token_per_frame (int): Number of tokens per frame in the input.
            vocab_size (int): Size of the input vocabulary.
            codebook_size (int): Size of the codec's codebook.
            codebook_dim (int): Dimension of the codec's codebook.
            pos (torch.nn.Module): Positional encoding layer.
            ln (torch.nn.LayerNorm): Layer normalization.
            decoder (torch.nn.Module): Decoder from the codec model.

        Args:
            input_size: Size of the input vocabulary.
            hf_model_tag (str): HuggingFace model tag for Espnet codec models.
            token_bias (int): The index of the first codec code.
            token_per_frame (int): Number of tokens per frame in the input.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding.
            positional_dropout_rate (float): Dropout rate after adding
                positional encoding.

        Raises:
            AssertionError: If the input tensor's dimensions or values are
                invalid.

        Examples:
            >>> codec_embedding = CodecEmbedding(input_size=400)
            >>> input_tensor = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
            >>> input_lengths = torch.tensor([4, 4])
            >>> output, output_lengths = codec_embedding(input_tensor, input_lengths)

        Note:
            The class uses an external model for codec inference and requires
            that the model be pre-trained and available through the specified
            HuggingFace model tag.
        """
        assert input.dim() == 2, input.size()
        assert input.size(1) % self.token_per_frame == 0, input.size()
        assert torch.all(input_lengths % self.token_per_frame == 0), input_lengths
        assert torch.all(input < self.vocab_size)

        B, Tnq = input.size()
        x = input.view(B, Tnq // self.token_per_frame, self.token_per_frame)
        x = x - self.token_bias

        for n in range(self.token_per_frame):
            x[:, :, n] -= n * self.codebook_size
        # NOTE (Jinchuan): do this clip so that the dequantization process
        # will not encounter an error. In practice, only the padding values
        # will exceed this range and is ignored by the length mask later.
        x = torch.clip(x, min=0, max=self.codebook_size - 1)

        z = self.quantizer.decode(x.permute(2, 0, 1)).permute(0, 2, 1)

        z = self.ln(z)
        z = self.pos(z)

        input_lengths = input_lengths // self.token_per_frame

        return z, input_lengths

    def output_size(self) -> int:
        """
            Return output length of feature dimension D, i.e. the embedding dim.

        This method provides the dimensionality of the output features generated
        by the embedding layer. This is particularly useful for understanding the
        size of the data that will be passed to subsequent layers in the neural
        network.

        Returns:
            int: The dimensionality of the output features, which is equal to
            the embedding dimension.

        Examples:
            >>> embedding = CodecEmbedding(input_size=400)
            >>> output_dim = embedding.output_size()
            >>> print(output_dim)
            400
        """
        return self.codebook_dim
