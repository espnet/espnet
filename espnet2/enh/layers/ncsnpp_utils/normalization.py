# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization layers."""
import functools

import torch
import torch.nn as nn


def get_normalization(config, conditional=False):
    """
    Obtain normalization modules from the config file.

    This function retrieves the appropriate normalization layer based on the
    provided configuration. It supports both conditional and non-conditional
    normalization layers.

    Args:
        config (dict): A configuration dictionary containing model parameters.
            It must have the key 'normalization' to specify the type of
            normalization to use.
        conditional (bool): A flag indicating whether to return a conditional
            normalization layer. Defaults to False.

    Returns:
        callable: A normalization layer class or a partial function for the
        specified normalization type.

    Raises:
        NotImplementedError: If a conditional normalization type is specified
            that is not implemented.
        ValueError: If the specified normalization type is unknown.

    Examples:
        >>> config = {'model': {'normalization': 'InstanceNorm'}}
        >>> normalization_layer = get_normalization(config)
        >>> print(normalization_layer)  # Output: <class 'torch.nn.modules.normalization.InstanceNorm2d'>

        >>> config = {'model': {'normalization': 'InstanceNorm++'}}
        >>> normalization_layer = get_normalization(config, conditional=True)
        >>> print(normalization_layer)  # Output: functools.partial object for ConditionalInstanceNorm2dPlus
    """
    norm = config.model.normalization
    if conditional:
        if norm == "InstanceNorm++":
            return functools.partial(
                ConditionalInstanceNorm2dPlus, num_classes=config.model.num_classes
            )
        else:
            raise NotImplementedError(f"{norm} not implemented yet.")
    else:
        if norm == "InstanceNorm":
            return nn.InstanceNorm2d
        elif norm == "InstanceNorm++":
            return InstanceNorm2dPlus
        elif norm == "VarianceNorm":
            return VarianceNorm2d
        elif norm == "GroupNorm":
            return nn.GroupNorm
        else:
            raise ValueError("Unknown normalization: %s" % norm)


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization layer.

    This layer applies batch normalization conditionally based on the class
    labels provided. It allows for learning different scaling and shifting
    parameters for different classes, which can be useful in tasks where
    the input data can be categorized into distinct classes.

    Attributes:
        num_features (int): Number of features (channels) in the input tensor.
        bias (bool): Whether to include bias parameters in the normalization.
        bn (nn.BatchNorm2d): Batch normalization layer without affine parameters.
        embed (nn.Embedding): Embedding layer to learn scaling and bias
            parameters based on class labels.

    Args:
        num_features (int): Number of features (channels) in the input tensor.
        num_classes (int): Number of classes for the conditional embedding.
        bias (bool): If True, includes bias in the normalization.

    Returns:
        Tensor: The normalized output tensor.

    Examples:
        >>> num_features = 16
        >>> num_classes = 10
        >>> batch_norm = ConditionalBatchNorm2d(num_features, num_classes)
        >>> x = torch.randn(8, num_features, 32, 32)  # Batch of 8 images
        >>> y = torch.randint(0, num_classes, (8,))  # Random class labels
        >>> output = batch_norm(x, y)

    Note:
        The scaling and bias parameters are initialized randomly. The
        scaling parameters are initialized to a normal distribution with
        mean 1 and standard deviation 0.02, while the bias parameters
        are initialized to zero.

    Raises:
        ValueError: If the input tensor dimensions do not match the expected
            shape for batch normalization.
    """

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[
                :, :num_features
            ].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        """
            Applies the conditional batch normalization to the input tensor.

        This method normalizes the input tensor `x` using batch normalization
        and applies a conditional scaling and shifting based on the class
        embeddings indexed by `y`. If bias is enabled, the method retrieves
        both scale (`gamma`) and bias (`beta`) parameters from the embedding
        layer and applies them to the normalized output.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W), where N
                is the batch size, C is the number of channels, H is the height,
                and W is the width.
            y (torch.Tensor): The tensor containing class indices of shape (N,)
                for which the normalization parameters will be conditioned.

        Returns:
            torch.Tensor: The output tensor after applying conditional batch
            normalization, having the same shape as the input tensor `x`.

        Examples:
            >>> c_bn = ConditionalBatchNorm2d(num_features=64, num_classes=10)
            >>> input_tensor = torch.randn(32, 64, 8, 8)  # Batch of 32 images
            >>> class_indices = torch.randint(0, 10, (32,))  # Random class indices
            >>> output_tensor = c_bn(input_tensor, class_indices)
            >>> output_tensor.shape
            torch.Size([32, 64, 8, 8])

        Note:
            The `y` tensor must contain valid class indices within the range
            [0, num_classes - 1].

        Raises:
            RuntimeError: If the input tensor `x` does not have the expected
            shape or if the class indices `y` are out of bounds.
        """
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm2d(nn.Module):
    """
    Applies Conditional Instance Normalization over a 4D input.

    ConditionalInstanceNorm2d normalizes the input tensor based on the
    instance statistics and the provided class condition. It uses an
    embedding layer to learn the scaling and shifting parameters
    conditioned on the class label.

    Attributes:
        num_features (int): Number of features (channels) in the input.
        bias (bool): If True, adds learnable bias to the output.
        instance_norm (nn.InstanceNorm2d): Instance normalization layer.
        embed (nn.Embedding): Embedding layer for class conditioning.

    Args:
        num_features (int): Number of features (channels) in the input.
        num_classes (int): Number of classes for the conditional embedding.
        bias (bool, optional): If True, adds learnable bias. Defaults to True.

    Returns:
        Tensor: The output tensor after applying conditional instance normalization.

    Examples:
        >>> import torch
        >>> norm = ConditionalInstanceNorm2d(num_features=3, num_classes=10)
        >>> x = torch.randn(5, 3, 32, 32)  # Batch of 5, 3 channels, 32x32
        >>> y = torch.randint(0, 10, (5,))  # Random class labels for batch
        >>> output = norm(x, y)
        >>> output.shape
        torch.Size([5, 3, 32, 32])

    Note:
        The input tensor should have the shape (N, C, H, W), where:
        N is the batch size, C is the number of channels, H is the height,
        and W is the width.

    Raises:
        RuntimeError: If the number of classes is less than or equal to zero.
    """

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[
                :, :num_features
            ].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        """
            Applies Conditional Instance Normalization to the input tensor.

        This method performs instance normalization on the input tensor `x`
        conditioned on the class labels `y`. If bias is enabled, it also applies
        learned scaling (gamma) and shifting (beta) parameters based on the
        class labels.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the
                batch size, C is the number of channels, H is the height, and W
                is the width.
            y (torch.Tensor): Class labels of shape (N,) where each entry is an
                integer representing the class index corresponding to the input
                tensor.

        Returns:
            torch.Tensor: The output tensor after applying conditional instance
            normalization. The output has the same shape as the input tensor.

        Examples:
            >>> model = ConditionalInstanceNorm2d(num_features=64, num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Random class labels
            >>> output = model(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])  # Output shape matches input shape
        """
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalVarianceNorm2d(nn.Module):
    """
    Conditional Variance Normalization layer.

    This layer normalizes the input tensor based on its variance,
    conditioned on class embeddings. The normalization is performed
    by scaling the input tensor using learnable parameters that are
    dependent on the class of the input.

    Attributes:
        num_features (int): The number of features (channels) in the input.
        bias (bool): A flag indicating whether to include bias in the
            normalization. If True, the layer will use an embedding
            to learn the scale.
        embed (nn.Embedding): An embedding layer to map class indices
            to scale parameters.

    Args:
        num_features (int): Number of features (channels) in the input tensor.
        num_classes (int): Number of classes for conditional normalization.
        bias (bool, optional): If True, include a bias term. Defaults to False.

    Returns:
        Tensor: The normalized output tensor.

    Examples:
        >>> import torch
        >>> model = ConditionalVarianceNorm2d(num_features=3, num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)  # Batch of 4, 3 channels, 32x32
        >>> y = torch.tensor([0, 1, 2, 3])  # Class indices for the batch
        >>> output = model(x, y)
        >>> output.shape
        torch.Size([4, 3, 32, 32])

    Note:
        The variance is computed across the spatial dimensions (height, width)
        of the input tensor. A small constant (1e-5) is added to the variance
        for numerical stability during the normalization.

    Raises:
        ValueError: If the input tensor does not have the expected number of
            dimensions.
    """

    def __init__(self, num_features, num_classes, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        """
            Apply the Conditional Variance Normalization to the input tensor.

        This method computes the variance of the input tensor `x` across
        the spatial dimensions (height and width), normalizes `x` using
        the computed variance, and scales the normalized tensor using the
        embeddings obtained from the class indices `y`. The embeddings are
        used to control the scaling factor for the normalized output.

        Args:
            x (torch.Tensor): The input tensor of shape
                (N, C, H, W), where N is the batch size, C is the number of
                channels, H is the height, and W is the width.
            y (torch.Tensor): The tensor of class indices of shape
                (N,) that determines the scaling factors for each input in
                the batch.

        Returns:
            torch.Tensor: The output tensor of shape (N, C, H, W) after
            applying the Conditional Variance Normalization.

        Examples:
            >>> import torch
            >>> model = ConditionalVarianceNorm2d(num_features=64, num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # A batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Random class indices for batch
            >>> output = model(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])

        Note:
            The variance is computed with a small constant added to avoid
            division by zero. The output will have the same shape as the
            input tensor.
        """
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class VarianceNorm2d(nn.Module):
    """
    Variance Normalization layer for 2D inputs.

    This layer normalizes the input tensor by its variance. It scales the
    normalized output using learnable parameters. This can help stabilize
    training and improve model performance by controlling the variance of
    the activations.

    Attributes:
        num_features (int): The number of input features (channels).
        bias (bool): If True, includes a bias term. Not currently used in
            this implementation.
        alpha (nn.Parameter): Learnable scaling parameter initialized from
            a normal distribution with mean 1 and standard deviation 0.02.

    Args:
        num_features (int): Number of features (channels) in the input.
        bias (bool, optional): Whether to include a bias term. Defaults to
            False.

    Returns:
        Tensor: The scaled and normalized output tensor.

    Examples:
        >>> variance_norm = VarianceNorm2d(num_features=64)
        >>> input_tensor = torch.randn(10, 64, 32, 32)  # Batch size of 10
        >>> output_tensor = variance_norm(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([10, 64, 32, 32])

    Note:
        The variance is computed over the spatial dimensions (height and
        width) of the input tensor. A small constant (1e-5) is added to
        avoid division by zero.
    """

    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        """
            Applies the variance normalization to the input tensor.

        This method computes the variance of the input tensor `x` along the spatial
        dimensions (height and width) and normalizes the input by dividing it by the
        square root of the variance (plus a small epsilon for numerical stability).
        It then scales the normalized output using the learnable parameter `alpha`.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where:
                N = batch size,
                C = number of channels,
                H = height of the feature map,
                W = width of the feature map.

        Returns:
            torch.Tensor: The normalized output tensor of the same shape as the input.

        Examples:
            >>> model = VarianceNorm2d(num_features=3)
            >>> input_tensor = torch.randn(2, 3, 4, 4)  # Batch size of 2, 3 channels
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([2, 3, 4, 4])

        Note:
            The normalization is performed per channel independently.
        """
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalNoneNorm2d(nn.Module):
    """
    Conditional None Normalization Layer.

    This layer applies a conditional normalization technique that allows
    the scaling of input features based on class embeddings. The output
    is computed by scaling the input tensor with learned parameters
    depending on the provided class index.

    Attributes:
        num_features (int): The number of features in the input tensor.
        bias (bool): A flag indicating whether to use a bias term.
        embed (nn.Embedding): An embedding layer that maps class indices to
            scaling parameters.

    Args:
        num_features (int): Number of input features (channels).
        num_classes (int): Number of classes for embedding.
        bias (bool, optional): If True, includes a bias term in the
            normalization. Defaults to True.

    Returns:
        Tensor: The normalized output tensor, scaled by the parameters
        derived from the input class index.

    Examples:
        >>> import torch
        >>> layer = ConditionalNoneNorm2d(num_features=64, num_classes=10)
        >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
        >>> y = torch.randint(0, 10, (8,))  # Random class indices for batch
        >>> output = layer(x, y)
        >>> print(output.shape)  # Should be (8, 64, 32, 32)

    Note:
        This layer is primarily used in scenarios where no specific
        normalization is desired but conditional scaling based on class
        information is still beneficial.

    Todo:
        - Implement additional normalization techniques as required.
    """

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[
                :, :num_features
            ].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        """
            Applies the Conditional None Normalization to the input tensor.

        This method normalizes the input tensor `x` based on the class
        embeddings obtained from the input tensor `y`. If the `bias`
        attribute is set to True, it uses two separate embeddings for scale
        (gamma) and bias (beta). If `bias` is False, it only applies the
        scale.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where
                N is the batch size, C is the number of features, H is the
                height, and W is the width.
            y (torch.Tensor): Class indices of shape (N,) used to index the
                embedding layer.

        Returns:
            torch.Tensor: The output tensor after applying Conditional None
            Normalization, with the same shape as the input tensor `x`.

        Examples:
            >>> model = ConditionalNoneNorm2d(num_features=64, num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Random class indices
            >>> output = model(x, y)
            >>> print(output.shape)  # Should be torch.Size([8, 64, 32, 32])

        Note:
            This normalization technique does not change the input tensor
            shape, and it is particularly useful for tasks where
            conditioning on class information is necessary.
        """
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * x
        return out


class NoneNorm2d(nn.Module):
    """
    A normalization layer that performs no normalization.

    This class implements a no-operation (identity) normalization layer
    for 2D inputs. It can be used as a placeholder in architectures where
    normalization is required but should be skipped.

    Attributes:
        num_features (int): The number of input features (channels).
        bias (bool): Whether to use bias in the layer. This is not used in
            this implementation.

    Args:
        num_features (int): The number of input features (channels).
        bias (bool): A flag to indicate if bias should be used (default: True).

    Returns:
        Tensor: The input tensor `x` is returned unchanged.

    Examples:
        >>> import torch
        >>> layer = NoneNorm2d(num_features=64)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 32, 32])  # Output shape is the same as input shape

    Note:
        This layer is primarily used in models where normalization is optional.
    """

    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        """
            Applies the normalization transformation to the input tensor.

        This method takes an input tensor `x` and a condition tensor `y` to
        perform normalization. The specific type of normalization applied
        depends on the initialization of the instance. If `bias` is enabled,
        the output will be adjusted with learned scaling (`gamma`) and
        shifting (`beta`) parameters derived from the embedding of `y`.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W), where
                N is the batch size, C is the number of channels, and
                H and W are the height and width of the input feature maps.
            y (torch.Tensor): The condition tensor of shape (N,) that contains
                class indices for the embedding layer. The indices must be in
                the range [0, num_classes).

        Returns:
            torch.Tensor: The output tensor after applying the normalization
            operation, with the same shape as the input tensor `x`.

        Examples:
            >>> model = ConditionalInstanceNorm2d(num_features=64, num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))   # Random class indices
            >>> output = model(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])

        Note:
            The output may differ depending on the values of `x` and `y`,
            as well as the learned parameters from the embedding layer.

        Raises:
            RuntimeError: If the input tensor `x` and condition tensor `y`
            have incompatible shapes or if any of the operations within
            the method fail.

        Todo:
            - Add support for additional normalization techniques.
            - Implement unit tests to validate the forward method.
        """
        return x


class InstanceNorm2dPlus(nn.Module):
    """
    Instance Normalization with additional parameters for improved performance.

    This layer performs instance normalization on input data and incorporates
    additional parameters to adapt the normalization based on the input's
    statistics. It is particularly useful for improving the stability and
    convergence of deep learning models.

    Attributes:
        num_features (int): The number of features (channels) in the input.
        bias (bool): Whether to include a bias term in the normalization.
        instance_norm (nn.InstanceNorm2d): The instance normalization layer.
        alpha (nn.Parameter): Learnable scaling parameter.
        gamma (nn.Parameter): Learnable scaling parameter.
        beta (nn.Parameter, optional): Learnable bias parameter, if `bias` is True.

    Args:
        num_features (int): Number of features (channels) in the input.
        bias (bool, optional): If True, includes a bias term. Defaults to True.

    Returns:
        Tensor: The normalized output tensor with the same shape as the input.

    Examples:
        >>> model = InstanceNorm2dPlus(num_features=64)
        >>> input_tensor = torch.randn(10, 64, 32, 32)  # Batch of 10 images
        >>> output_tensor = model(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 64, 32, 32])

    Note:
        The normalization is performed per-instance, meaning that each
        instance in the batch is normalized independently.

    Raises:
        RuntimeError: If the input tensor is not of the expected shape.

    Todo:
        Consider extending the functionality to include other normalization
        techniques in future iterations.
    """

    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """
            Applies the Conditional Instance Normalization operation.

        This method normalizes the input tensor `x` using instance normalization,
        followed by a conditional scaling and shifting based on the class
        embeddings obtained from the input `y`. The normalization is computed
        using the statistics of the input tensor, and the scaling and shifting
        parameters are learned through the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the
                batch size, C is the number of channels, H is the height, and W
                is the width.
            y (torch.Tensor): Class labels tensor of shape (N,) containing
                class indices corresponding to each input in the batch.

        Returns:
            torch.Tensor: The normalized output tensor with the same shape as
                input `x`.

        Examples:
            >>> model = ConditionalInstanceNorm2dPlus(num_features=64, num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Random class indices for the batch
            >>> output = model(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])

        Note:
            The instance normalization is computed over the spatial dimensions of
            the input tensor, and the class embeddings are used to adjust the
            output through learned parameters.
        """
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalInstanceNorm2dPlus(nn.Module):
    """
    Conditional Instance Normalization layer with additional scaling and bias.

    This class implements a conditional instance normalization layer that
    normalizes input features using learned parameters based on the provided
    class labels. The layer includes the capability to apply an additional
    scaling factor and a bias term for more flexible output.

    Attributes:
        num_features (int): The number of input features (channels).
        bias (bool): Indicates whether to include a bias term in the layer.
        instance_norm (nn.InstanceNorm2d): Instance normalization module.
        embed (nn.Embedding): Embedding layer for learning scaling and bias.

    Args:
        num_features (int): Number of input features (channels).
        num_classes (int): Number of classes for the conditional embedding.
        bias (bool): If True, includes a bias term (default: True).

    Returns:
        Tensor: Normalized output tensor.

    Examples:
        >>> layer = ConditionalInstanceNorm2dPlus(num_features=64, num_classes=10)
        >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
        >>> y = torch.randint(0, 10, (8,))  # Random class labels for the batch
        >>> output = layer(x, y)

    Note:
        The input tensor `x` is expected to have the shape
        (batch_size, num_features, height, width). The class labels `y`
        should have the shape (batch_size,).

    Todo:
        - Add support for additional normalization techniques.
    """

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, : 2 * num_features].normal_(
                1, 0.02
            )  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[
                :, 2 * num_features :
            ].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        """
            Applies the Conditional Instance Normalization operation.

        This method performs Conditional Instance Normalization on the input tensor
        `x` based on the class label `y`. The operation normalizes the input tensor
        using the instance normalization technique and applies learnable scaling
        and shifting parameters based on the class embedding.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W) where:
                - N is the batch size,
                - C is the number of channels,
                - H is the height,
                - W is the width.
            y (torch.Tensor): The class label tensor of shape (N,) where each element
                corresponds to the class index for the respective input in `x`.

        Returns:
            torch.Tensor: The output tensor after applying Conditional Instance
            Normalization, with the same shape as the input tensor `x`.

        Examples:
            >>> norm_layer = ConditionalInstanceNorm2dPlus(num_features=64, num_classes=10)
            >>> input_tensor = torch.randn(8, 64, 32, 32)  # Example input
            >>> class_labels = torch.randint(0, 10, (8,))  # Random class labels
            >>> output_tensor = norm_layer(input_tensor, class_labels)
            >>> print(output_tensor.shape)  # Should be [8, 64, 32, 32]

        Note:
            The input tensor `x` must have 4 dimensions, and the class label `y`
            must be a 1-dimensional tensor with class indices ranging from 0 to
            num_classes - 1.

        Raises:
            IndexError: If the class index in `y` is out of bounds for the embedding
            layer.
        """
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out
