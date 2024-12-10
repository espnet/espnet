# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec
# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from espnet2.gan_codec.shared.quantizer.modules.distrib import broadcast_tensors


def default(val: Any, d: Any) -> Any:
    """
        Default function to return a value if it is not None, otherwise return a default.

    Args:
        val (Any): The value to check.
        d (Any): The default value to return if val is None.

    Returns:
        Any: Returns val if it is not None; otherwise, returns d.

    Examples:
        >>> default(None, 42)
        42
        >>> default(10, 42)
        10
    """
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """
    Update the moving average in-place using exponential decay.

    This function updates a moving average tensor in-place by applying an
    exponential decay to the existing moving average and adding a new value.
    It is useful for maintaining a running average that gives more weight
    to recent values.

    Args:
        moving_avg (torch.Tensor): The tensor containing the current moving
            average, which will be updated in-place.
        new (torch.Tensor): The new value to be incorporated into the
            moving average.
        decay (float): The decay factor used to compute the weighted average.
            It should be in the range [0, 1), where values closer to 1 give
            more weight to the previous average and values closer to 0 give
            more weight to the new value.

    Examples:
        >>> moving_avg = torch.tensor([0.0])
        >>> new_value = torch.tensor([1.0])
        >>> decay_factor = 0.9
        >>> ema_inplace(moving_avg, new_value, decay_factor)
        >>> print(moving_avg)  # Output: tensor([0.1])

        >>> moving_avg = torch.tensor([0.5])
        >>> new_value = torch.tensor([2.0])
        >>> decay_factor = 0.8
        >>> ema_inplace(moving_avg, new_value, decay_factor)
        >>> print(moving_avg)  # Output: tensor([1.1])
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    """
    Apply Laplace smoothing to a given tensor.

    This function performs Laplace smoothing on the input tensor `x` to
    prevent zero probabilities in categorical distributions. The
    smoothing technique adds a small constant (epsilon) to each element
    of `x` and normalizes by the total sum adjusted for the number of
    categories.

    Args:
        x (Tensor): The input tensor representing counts or frequencies
            for each category.
        n_categories (int): The total number of categories to be
            considered for smoothing.
        epsilon (float, optional): A small value added to each count
            for numerical stability. Defaults to 1e-5.

    Returns:
        Tensor: A tensor of the same shape as `x`, containing the
        smoothed probabilities.

    Examples:
        >>> import torch
        >>> counts = torch.tensor([0, 2, 3])
        >>> smoothed_probs = laplace_smoothing(counts, n_categories=3)
        >>> print(smoothed_probs)
        tensor([0.1667, 0.3333, 0.5000])

    Note:
        This function is particularly useful in scenarios involving
        categorical data where some categories may not have any
        observations, thereby resulting in zero probabilities.
    """
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    """
        Initialize a tensor with values drawn from a uniform distribution.

    This function initializes a tensor with the specified shape using the
    Kaiming uniform initialization method. This method is often used to set
    the initial weights of neural networks to help with convergence during
    training.

    Args:
        *shape (int): The desired shape of the output tensor.

    Returns:
        torch.Tensor: A tensor of the specified shape initialized using the
        Kaiming uniform method.

    Examples:
        >>> tensor = uniform_init(3, 4)
        >>> tensor.shape
        torch.Size([3, 4])

        >>> tensor = uniform_init(2, 2, 2)
        >>> tensor.shape
        torch.Size([2, 2, 2])
    """
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    """
        Samples a specified number of vectors from the given samples.

    This function selects `num` random vectors from the input tensor `samples`. If
    the number of available samples is greater than or equal to `num`, it selects
    random indices without replacement. If there are fewer available samples than
    `num`, it selects indices with replacement.

    Args:
        samples (Tensor): A tensor of shape (N, D) where N is the number of samples
                          and D is the dimensionality of each sample.
        num (int): The number of vectors to sample from the input tensor.

    Returns:
        Tensor: A tensor containing the sampled vectors of shape (num, D).

    Examples:
        >>> samples = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> sampled = sample_vectors(samples, 2)
        >>> print(sampled)
        tensor([[3.0, 4.0], [1.0, 2.0]])  # Output may vary due to randomness.

        >>> sampled = sample_vectors(samples, 5)
        >>> print(sampled)
        tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [5.0, 6.0]])
        # Output may vary due to randomness.
    """
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """
    Perform k-means clustering on the given samples.

    This function applies the k-means algorithm to cluster the input
    samples into a specified number of clusters. The clustering is done
    iteratively, updating the cluster centroids based on the assigned
    samples in each iteration.

    Args:
        samples (Tensor): Input tensor of shape (N, D) where N is the number
            of samples and D is the dimension of each sample.
        num_clusters (int): The number of clusters to form.
        num_iters (int, optional): The number of iterations to run the k-means
            algorithm. Defaults to 10.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Tensor: The final cluster centroids of shape (num_clusters, D).
            - Tensor: The number of samples assigned to each cluster, of shape
              (num_clusters,).

    Examples:
        >>> import torch
        >>> samples = torch.rand(100, 2)  # 100 samples in 2D
        >>> centroids, cluster_sizes = kmeans(samples, num_clusters=3, num_iters=5)
        >>> print(centroids.shape)  # Should print: torch.Size([3, 2])
        >>> print(cluster_sizes.shape)  # Should print: torch.Size([3])

    Note:
        The input samples must be a 2D tensor. The algorithm may not converge
        in cases where the number of clusters is too large or the data is not
        well-separated.
    """
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    # Cluster centroids and number of frames per cluster
    return means, bins


class EuclideanCodebook(nn.Module):
    """
    Codebook with Euclidean distance for vector quantization.

    This class implements a codebook that utilizes Euclidean distance for
    quantizing input vectors. It supports k-means initialization, decay for
    exponential moving average (EMA), and expiration of dead codes based on
    cluster sizes.

    Attributes:
        decay (float): Decay factor for EMA updates of the codebooks.
        codebook_size (int): The size of the codebook.
        kmeans_iters (int): Number of iterations for the k-means algorithm.
        epsilon (float): Small value for numerical stability during calculations.
        threshold_ema_dead_code (int): Minimum cluster size for expiration of codes.

    Args:
        dim (int): Dimension of the input vectors.
        codebook_size (int): Number of code vectors in the codebook.
        kmeans_init (bool): If True, uses k-means for initializing the codebook.
        kmeans_iters (int): Number of iterations for k-means initialization.
        decay (float): Decay factor for EMA updates.
        epsilon (float): Small value for numerical stability.
        threshold_ema_dead_code (int): Minimum size threshold for dead code expiration.

    Examples:
        >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
        >>> x = torch.randn(10, 128)  # 10 vectors of dimension 128
        >>> embed_indices = codebook.encode(x)
        >>> quantized_output = codebook.decode(embed_indices)

    Note:
        The k-means initialization is performed only on the first batch of
        training data if kmeans_init is set to True. The class maintains
        an exponential moving average of the codebook vectors and their
        corresponding cluster sizes.

    Raises:
        ValueError: If `dim` or `codebook_size` is less than or equal to zero.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., torch.Tensor], Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """
        Initialize the codebook embeddings using k-means clustering.

        This method initializes the codebook embeddings if they have not
        been initialized yet. It performs k-means clustering on the input
        data to determine the initial codebook vectors and updates the
        relevant buffers.

        Args:
            data (Tensor): The input data used for initializing the codebook.
                The shape of the tensor should be (N, D), where N is the
                number of samples and D is the dimension of each sample.

        Returns:
            None: This method modifies the internal state of the class in place.

        Note:
            This method is only activated during the first call, as it checks
            the `inited` flag to determine whether the embeddings have already
            been initialized. If they have, the method returns early without
            making any changes.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> data = torch.randn(1000, 128)
            >>> codebook.init_embed_(data)  # Initializes the codebook embeddings

        Raises:
            ValueError: If the input data is not of the expected shape.
        """
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        """
            Replace the entries in the codebook based on a mask.

        This method updates the codebook entries with randomly selected vectors
        from the provided samples where the corresponding mask is True. The
        existing codebook entries remain unchanged where the mask is False.

        Args:
            samples (Tensor): The tensor containing sample vectors from which
                new codebook entries will be drawn. Shape should be (N, D)
                where N is the number of samples and D is the dimensionality
                of each sample.
            mask (Tensor): A boolean tensor indicating which codebook entries
                should be replaced. Shape should be (C,) where C is the number
                of codebook entries.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> samples = torch.randn(10, 128)  # 10 samples of dimension 128
            >>> mask = torch.tensor([True, False, True, False, True,
            ...                      False, True, False, True, False,
            ...                      True, False, True, False, True,
            ...                      False, True, False, True, False,
            ...                      True, False, True, False, True,
            ...                      False, True, False, True, False])
            >>> codebook.replace_(samples, mask)

        Note:
            This method is typically used in the training process to adapt
            the codebook entries based on the current batch of samples.
        """
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        """
            Expire codes that have a low exponential moving average cluster size.

        This method checks the cluster sizes and replaces any codes in the codebook
        that have an exponential moving average size less than the specified
        threshold (`threshold_ema_dead_code`) with randomly selected vectors
        from the current batch. This helps to ensure that the codebook remains
        relevant and does not contain unused or under-utilized codes.

        Args:
            batch_samples (Tensor): A tensor containing the current batch of samples.
                This should be a 2D tensor where the first dimension is the number
                of samples and the second dimension is the feature size.

        Returns:
            None: This method modifies the codebook in place and does not return
            any value.

        Raises:
            None: This method does not raise any exceptions.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> batch_samples = torch.randn(32, 128)  # 32 samples, 128 features
            >>> codebook.expire_codes_(batch_samples)

        Note:
            - The method only executes the expiration logic if
              `threshold_ema_dead_code` is greater than 0.
            - If no codes are expired, the method returns early without making
              any changes to the codebook.
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        """
                Preprocess input data for the Euclidean codebook.

        This function rearranges the input tensor `x` to ensure it has the correct shape
        for further processing within the codebook. Specifically, it flattens the last
        dimension while maintaining the rest of the structure.

        Args:
            x (torch.Tensor): The input tensor to preprocess. It can have any number
                of leading dimensions, with the last dimension representing the features.

        Returns:
            torch.Tensor: The preprocessed tensor with the last dimension flattened
                into a single dimension while retaining the leading dimensions.

        Examples:
            >>> import torch
            >>> x = torch.randn(2, 3, 4)  # A tensor with shape (2, 3, 4)
            >>> preprocessed_x = preprocess(x)
            >>> print(preprocessed_x.shape)  # Should output: torch.Size([6, 4])
        """
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        """
                Core vector quantization implementation for the EuclideanCodebook package.

        This module provides an implementation of vector quantization using
        Euclidean distance, including the ability to initialize codebooks using
        k-means clustering, and manage codebook updates using exponential moving
        averages.

        Attributes:
            dim (int): Dimension of the input vectors.
            codebook_size (int): Size of the codebook (number of centroids).
            kmeans_init (bool): Flag indicating whether to initialize the codebook
                with k-means.
            kmeans_iters (int): Number of iterations for k-means initialization.
            decay (float): Decay rate for the exponential moving average of the
                codebooks.
            epsilon (float): Small value for numerical stability during calculations.
            threshold_ema_dead_code (int): Minimum cluster size for a code to be
                considered alive.

        Args:
            x (torch.Tensor): Input tensor for quantization of shape (B, T, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Quantized output of shape (B, T, D).
                - Codebook index of shape (B, T).

        Raises:
            ValueError: If the input tensor does not have the expected shape.

        Examples:
            # Example of using the EuclideanCodebook
            codebook = EuclideanCodebook(dim=128, codebook_size=256, kmeans_init=True)
            input_tensor = torch.randn(10, 20, 128)  # Batch of 10, 20 time steps, 128 dimensions
            quantized_output, codebook_indices = codebook(input_tensor)

        Note:
            The forward method initializes the codebook only on the first call
            using the provided input data.

        Todo:
            - Implement more efficient k-means initialization.
            - Optimize the quantization process for larger codebooks.
        """
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        """
            Post-process the embedding indices to reshape them into the desired output shape.

        This method takes the embedding indices produced by the quantization process
        and reshapes them according to the specified shape. It is particularly useful
        in maintaining the original dimensions of the input tensor after quantization.

        Args:
            embed_ind (torch.Tensor): The embedding indices obtained from the quantization
                process, typically a tensor of shape (B, T) where B is the batch size and
                T is the number of time steps.
            shape (tuple): The desired output shape for the reshaped embedding indices.
                This should generally correspond to the shape of the original input tensor
                before quantization.

        Returns:
            torch.Tensor: The reshaped embedding indices with the specified shape.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> embed_ind = torch.tensor([[0, 1, 2], [3, 4, 5]])
            >>> shape = (2, 3)  # Batch size of 2 and 3 time steps
            >>> reshaped_embed_ind = codebook.postprocess_emb(embed_ind, shape)
            >>> print(reshaped_embed_ind.shape)
            torch.Size([2, 3])  # Output shape matches the specified shape
        """
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        """
        Dequantizes the given embedding indices using the codebook.

        This method retrieves the corresponding vectors from the codebook
        based on the provided embedding indices. It effectively maps the
        quantized indices back to their respective continuous representations.

        Args:
            embed_ind (Tensor): A tensor containing the indices of the
                quantized embeddings. The shape should be (B, T) where
                B is the batch size and T is the number of time steps.

        Returns:
            Tensor: A tensor containing the dequantized vectors. The shape
            will be (B, T, D), where D is the dimension of the codebook
            vectors.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> indices = torch.tensor([[0, 1], [2, 3]])
            >>> dequantized_vectors = codebook.dequantize(indices)
            >>> print(dequantized_vectors.shape)
            torch.Size([2, 2, 128])
        """
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        """
                Encodes input tensor using the Euclidean codebook for vector quantization.

        This method preprocesses the input tensor, quantizes it by finding the nearest
        codebook vectors, and then post-processes the resulting indices to match the
        original input shape.

        Args:
            x (Tensor): Input tensor of shape (B, T, D) where B is the batch size,
                        T is the sequence length, and D is the dimension of each vector.

        Returns:
            Tensor: The encoded indices of shape (B, T) that correspond to the
                    nearest codebook vectors.

        Examples:
            >>> codebook = EuclideanCodebook(dim=256, codebook_size=512)
            >>> input_tensor = torch.randn(10, 20, 256)  # Batch of 10, 20 time steps, 256 dimensions
            >>> encoded_indices = codebook.encode(input_tensor)
            >>> print(encoded_indices.shape)  # Output: torch.Size([10, 20])
        """
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        """
                Decode the indices of the embedded vectors into their original representations.

        This method takes the encoded indices (embedded indices) and retrieves the
        corresponding vectors from the codebook. It effectively reverses the quantization
        process, allowing you to obtain the actual vector representation for further
        processing or analysis.

        Args:
            embed_ind (Tensor): A tensor containing the indices of the embeddings to be
                decoded. The shape should be (B, T) where B is the batch size and T is
                the number of time steps.

        Returns:
            Tensor: A tensor containing the decoded vectors corresponding to the input
                indices. The shape will be (B, T, D) where D is the dimensionality of the
                vectors in the codebook.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=512)
            >>> indices = torch.tensor([[1, 2], [3, 4]])
            >>> decoded_vectors = codebook.decode(indices)
            >>> print(decoded_vectors.shape)
            torch.Size([2, 2, 128])  # Assuming the codebook vectors are of dimension 128

        Note:
            The decoded vectors are obtained using an embedding lookup on the codebook.
        """
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        """
        Codebook Forward with EMA.

        This method processes the input tensor `x` through the codebook
        using exponential moving average (EMA) for quantization. It
        initializes the embedding if it is the first forward pass,
        performs quantization, and handles the training updates for
        the codebook.

        Args:
            x (Tensor): Vector for quantization (B, T, D), where B is the
                batch size, T is the sequence length, and D is the
                dimension of the vectors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Tensor: Quantized output (B, T, D)
                - Tensor: Codebook Index (B, T)

        Examples:
            >>> codebook = EuclideanCodebook(dim=256, codebook_size=512)
            >>> input_tensor = torch.randn(8, 10, 256)  # Batch of 8, seq len 10
            >>> quantized_output, codebook_indices = codebook(input_tensor)
            >>> print(quantized_output.shape)  # Should output: torch.Size([8, 10, 256])
            >>> print(codebook_indices.shape)   # Should output: torch.Size([8, 10])

        Note:
            The method will only initialize the embedding on the first
            forward call, which is controlled by the `inited` buffer.
            If `self.training` is True, it updates the codebook with
            the current batch statistics.

        Raises:
            RuntimeError: If the input tensor does not have the expected
                shape or is not a 3D tensor.
        """
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)  # (BxT, D)

        # Initialize the embedding (only activated for the first time)
        self.init_embed_(x)

        # Quantization Process
        embed_ind = self.quantize(x)  # (BxT)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # (BxT, V)
        embed_ind = self.postprocess_emb(embed_ind, shape)  # (B, T)
        quantize = self.dequantize(embed_ind)  # (B, T, D)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)

            # ema update number of frames per cluster
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)

            # Use encoder embedding to update ema with assignments
            embed_sum = x.t() @ embed_onehot  # (D, BxT) @ (BxT, V) -> (D, V)

            # ema udpate embedding
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """
    Vector quantization implementation.

    Currently supports only Euclidean distance.

    Args:
        dim (int): Dimension of the input data.
        codebook_size (int): Size of the codebook.
        codebook_dim (Optional[int]): Dimension of the codebook. If not defined,
            uses the specified dimension in `dim`.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for k-means initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster size
            less than the specified threshold with a randomly selected vector from
            the current batch.
        commitment_weight (float): Weight for the commitment loss during training.
        quantizer_dropout (bool): Whether to apply dropout to the quantizer.

    Examples:
        >>> vq = VectorQuantization(dim=256, codebook_size=512)
        >>> x = torch.randn(10, 256)  # Batch of 10 samples with 256 features
        >>> quantized_output, embed_indices, loss = vq(x)

    Note:
        The forward method returns different outputs depending on whether the model
        is in training or evaluation mode. In training mode, it returns the
        quantized output, embedding indices, and losses; while in evaluation mode,
        it returns only the quantized output and embedding indices.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

    @property
    def codebook(self):
        """
                Core vector quantization implementation.

        This module provides a framework for vector quantization, including an
        implementation of Euclidean distance-based codebooks and the vector
        quantization algorithm. The main classes included are `EuclideanCodebook`,
        `VectorQuantization`, and `ResidualVectorQuantization`, each with specific
        functions to handle encoding, decoding, and loss calculations.

        Attributes:
            None

        Args:
            dim (int): Dimension of the input vectors.
            codebook_size (int): Size of the codebook.
            codebook_dim (Optional[int]): Dimension of the codebook. If not defined,
                uses the specified dimension in `dim`.
            decay (float): Decay for exponential moving average over the codebooks.
            epsilon (float): Epsilon value for numerical stability.
            kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            kmeans_iters (int): Number of iterations used for k-means initialization.
            threshold_ema_dead_code (int): Threshold for dead code expiration.
                Replace any codes that have an exponential moving average cluster size
                less than the specified threshold with a randomly selected vector from
                the current batch.
            commitment_weight (float): Weight for commitment loss in vector
                quantization.
            quantizer_dropout (bool): Flag to enable dropout in quantization layers.

        Examples:
            # Initialize a vector quantization model
            vq_model = VectorQuantization(dim=256, codebook_size=512)

            # Forward pass with input tensor
            quantized_output, indices, loss = vq_model(input_tensor)

            # Encode and decode process
            encoded_indices = vq_model.encode(input_tensor)
            reconstructed_tensor = vq_model.decode(encoded_indices)

        Note:
            The code is inspired by implementations from the `encodec` and
            `vector-quantize-pytorch` repositories. The code is licensed under the
            Apache 2.0 and MIT licenses.

        Todo:
            - Implement support for other distance metrics in vector quantization.
            - Optimize the k-means initialization process for larger datasets.
        """
        return self._codebook.embed

    def encode(self, x):
        """
            Encodes the input tensor into indices of the codebook.

        This method preprocesses the input tensor, quantizes it by finding the
        nearest codebook entries, and then post-processes the indices to match
        the original shape.

        Args:
            x (Tensor): Input tensor to be encoded. The expected shape is
                        (B, D, N) where B is the batch size, D is the dimension
                        of each vector, and N is the number of vectors.

        Returns:
            Tensor: A tensor of shape (B, N) containing the indices of the
                    quantized vectors from the codebook.

        Examples:
            >>> vq = VectorQuantization(dim=256, codebook_size=512)
            >>> input_tensor = torch.randn(2, 256, 10)  # Batch of 2, 10 vectors
            >>> indices = vq.encode(input_tensor)
            >>> print(indices.shape)  # Output: torch.Size([2, 10])
        """
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        """
            Decode the quantized indices back to their original vector representations.

        This method takes the indices of the quantized vectors and retrieves the
        corresponding vectors from the codebook, effectively reversing the
        quantization process.

        Args:
            embed_ind (Tensor): A tensor containing the indices of the quantized
                vectors. It is expected to be of shape (B, T), where B is the
                batch size and T is the number of time steps.

        Returns:
            Tensor: A tensor of shape (B, T, D) containing the decoded vectors,
                where D is the dimension of the original vectors.

        Examples:
            >>> quantizer = VectorQuantization(dim=128, codebook_size=256)
            >>> indices = torch.randint(0, 256, (10, 20))  # Simulated indices
            >>> decoded_vectors = quantizer.decode(indices)
            >>> print(decoded_vectors.shape)  # Output: torch.Size([10, 20, 128])
        """
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, mask=None):
        """
        Codebook Forward with EMA.

        This method performs the forward pass of the vector quantization
        process, applying exponential moving average (EMA) updates to the
        codebook embeddings during training.

        Args:
            x (Tensor): Vector for quantization with shape (B, T, D),
                where B is the batch size, T is the sequence length,
                and D is the dimension of the vectors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Tensor: Quantized output with shape (B, T, D).
                - Tensor: Codebook Index with shape (B, T).

        Note:
            During the training phase, the method updates the codebook
            using EMA and expires codes based on their usage.

        Examples:
            >>> model = VectorQuantization(dim=128, codebook_size=512)
            >>> input_tensor = torch.randn(10, 20, 128)  # Batch of 10, 20 time steps
            >>> quantized_output, codebook_index = model(input_tensor)
        """
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        if not self.quantizer_dropout:
            loss = torch.tensor([0.0], device=device, requires_grad=self.training)

            if self.training:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, loss
        else:
            commit_loss = torch.tensor(
                [0.0], device=device, requires_grad=self.training
            )
            quant_loss = torch.tensor([0.0], device=device, requires_grad=self.training)
            if self.training:
                if self.quantizer_dropout:
                    _commit_loss = F.mse_loss(
                        quantize.detach(), x, reduction="none"
                    ).mean([1, 2])
                    commit_loss = commit_loss + (_commit_loss * mask).mean()
                    _quant_loss = F.mse_loss(
                        quantize, x.detach(), reduction="none"
                    ).mean([1, 2])
                    quant_loss = quant_loss + (_quant_loss * mask).mean()

                else:
                    _commit_loss = F.mse_loss(quantize.detach(), x)
                    commit_loss = commit_loss + _commit_loss
                    _quant_loss = F.mse_loss(quantize, x.detach(), reduction="none")
                    quant_loss = quant_loss + _quant_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, commit_loss, quant_loss


class ResidualVectorQuantization(nn.Module):
    """
    Residual vector quantization implementation.

    Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf.

    This class implements a residual vector quantization mechanism that allows
    for more efficient encoding by utilizing multiple quantizers in sequence.
    Each quantizer processes the residual from the previous step, effectively
    reducing the quantization error iteratively.

    Attributes:
        layers (nn.ModuleList): A list of VectorQuantization layers.
        quantizer_dropout (bool): Whether to apply dropout to quantizers during
            training.

    Args:
        num_quantizers (int): The number of quantizers to be used in the
            residual vector quantization process.
        **kwargs: Additional keyword arguments passed to the VectorQuantization
            initialization.

    Returns:
        quantized_out (Tensor): The final quantized output.
        out_indices (Tensor): Indices of the quantized representations.
        out_losses (Tensor): Losses associated with each quantization step
            during training.

    Examples:
        >>> rvq = ResidualVectorQuantization(num_quantizers=3, dim=128,
        ...                                   codebook_size=256)
        >>> x = torch.randn(16, 10, 128)  # Batch of 16 sequences of length 10
        >>> quantized_out, indices, losses = rvq(x)

    Note:
        The implementation includes dropout behavior for the quantizers, which
        can be controlled via the `quantizer_dropout` parameter. This allows
        for random selection of quantizers during training, improving model
        robustness.

    Raises:
        ValueError: If `num_quantizers` is not a positive integer.
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.quantizer_dropout = kwargs.get("quantizer_dropout")

    def forward(self, x, n_q: Optional[int] = None):
        """
        Perform forward pass for residual vector quantization.

        This method processes the input tensor through a series of vector
        quantization layers, computing the quantized output, indices, and
        losses associated with the quantization process.

        Args:
            x (Tensor): Input tensor to be quantized (B, D, N).
            n_q (Optional[int]): Number of quantizers to use. If None,
                uses all available quantizers.

        Returns:
            Tensor: Quantized output (B, D, N).
            List[Tensor]: Indices of the quantized vectors for each layer.
            List[Tensor]: Losses associated with each layer's quantization
                (if not using dropout).

        Note:
            If `quantizer_dropout` is enabled, only a subset of quantizers
            will be used during training, based on a dropout mechanism.

        Examples:
            >>> rvq = ResidualVectorQuantization(num_quantizers=3,
            ...                                    dim=128,
            ...                                    codebook_size=256)
            >>> input_tensor = torch.randn(10, 128, 20)
            >>> quantized_output, indices, losses = rvq(input_tensor)
        """
        quantized_out = 0.0
        residual = x

        if not self.quantizer_dropout:
            all_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)

            for layer in self.layers[:n_q]:
                quantized, indices, loss = layer(residual)
                residual = residual - quantized
                quantized_out = quantized_out + quantized

                all_indices.append(indices)
                all_losses.append(loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
            return quantized_out, out_indices, out_losses
        else:
            all_commit_losses = []
            all_quant_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)
            if self.training:
                n_q = torch.ones((x.shape[0],)) * len(self.layers) + 1
                dropout = torch.randint(1, len(self.layers) + 1, (x.shape[0],))
                n_dropout = int(x.shape[0] * self.quantizer_dropout)
                n_q[:n_dropout] = dropout[:n_dropout]
                n_q = n_q.to(x.device)

            for i, layer in enumerate(self.layers):
                if self.training is False and i >= n_q:
                    break
                mask = torch.full((x.shape[0],), fill_value=i, device=x.device) < n_q
                quantized, indices, commit_loss, quant_loss = layer(residual, mask)
                residual = residual - quantized
                quantized_out = quantized_out + quantized * mask[:, None, None]

                all_indices.append(indices)
                all_commit_losses.append(commit_loss)
                all_quant_losses.append(quant_loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_commit_losses, out_quant_losses, out_indices = map(
                torch.stack, (all_commit_losses, all_quant_losses, all_indices)
            )
            return quantized_out, out_indices, out_commit_losses, out_quant_losses

    def encode(
        self, x: torch.Tensor, n_q: Optional[int] = None, st: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encodes input tensor using the vector quantization process.

        This method preprocesses the input tensor, performs quantization,
        and post-processes the resulting indices. The output is a tensor
        containing the indices of the quantized vectors.

        Args:
            x (Tensor): Input tensor of shape (B, T, D), where B is the batch
                size, T is the sequence length, and D is the dimensionality
                of the input vectors.

        Returns:
            Tensor: A tensor of shape (B, T) containing the indices of the
                quantized vectors.

        Examples:
            >>> import torch
            >>> vq = VectorQuantization(dim=128, codebook_size=256)
            >>> input_tensor = torch.randn(10, 20, 128)  # (B, T, D)
            >>> indices = vq.encode(input_tensor)
            >>> print(indices.shape)  # Output: torch.Size([10, 20])
        """
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """
            Decode the quantized indices into their corresponding vectors.

        This method retrieves the original vectors from the codebook using the
        provided quantized indices. It effectively performs the reverse operation
        of the quantization process.

        Args:
            embed_ind (Tensor): A tensor of quantized indices of shape (B, T)
                where B is the batch size and T is the number of time steps.

        Returns:
            Tensor: A tensor containing the decoded vectors of shape (B, T, D),
                where D is the dimension of the vectors.

        Examples:
            >>> codebook = EuclideanCodebook(dim=128, codebook_size=256)
            >>> indices = torch.tensor([[0, 1], [2, 3]])  # Example indices
            >>> decoded_vectors = codebook.decode(indices)
            >>> print(decoded_vectors.shape)  # Should output: torch.Size([2, 2, 128])
        """
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
