# noqa E501: Ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/src/models/adapt_layers.py
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from functools import partial

import torch
import torch.nn as nn


def make_adapt_layer(type, indim, enrolldim, ninputs=1):
    """
        Creates an adaptation layer based on the specified type.

    This function returns an instance of an adaptation layer class
    defined in the `adaptation_layer_types` dictionary. The type of
    layer to create is specified by the `type` argument. The input
    dimensions and enrollment dimensions must also be provided.

    Attributes:
        type (str): The type of adaptation layer to create.
                    Possible values are "concat", "muladd", and "mul".
        indim (int): The input dimension for the adaptation layer.
        enrolldim (int): The enrollment dimension for the adaptation layer.
        ninputs (int, optional): The number of input tensors to the layer.
                                 Defaults to 1.

    Args:
        type (str): The type of adaptation layer to create.
        indim (int): The input dimension for the adaptation layer.
        enrolldim (int): The enrollment dimension for the adaptation layer.
        ninputs (int, optional): The number of input tensors to the layer.
                                 Defaults to 1.

    Returns:
        nn.Module: An instance of the specified adaptation layer class.

    Raises:
        KeyError: If the specified type does not exist in
                   `adaptation_layer_types`.

    Examples:
        >>> layer = make_adapt_layer("concat", 128, 256)
        >>> print(layer)
        ConcatAdaptLayer(...)

        >>> layer = make_adapt_layer("muladd", 128, 256, ninputs=2)
        >>> print(layer)
        MulAddAdaptLayer(...)
    """
    adapt_class = adaptation_layer_types.get(type)
    return adapt_class(indim, enrolldim, ninputs)


def into_tuple(x):
    """
    Transforms a tensor, list, or tuple into a tuple.

    This function checks the type of the input `x` and converts it into a
    tuple. If `x` is a list, it will be converted to a tuple containing
    the same elements. If `x` is a tensor, it will be wrapped in a tuple.
    If `x` is already a tuple, it will be returned as is. If `x` is of
    an unsupported type, a ValueError will be raised.

    Args:
        x: A tensor, list, or tuple to be converted into a tuple.

    Returns:
        tuple: A tuple representation of the input `x`.

    Raises:
        ValueError: If `x` is neither a tensor, list, nor tuple.

    Examples:
        >>> into_tuple([1, 2, 3])
        (1, 2, 3)

        >>> into_tuple(torch.tensor([1, 2, 3]))
        (tensor([1, 2, 3]),)

        >>> into_tuple((1, 2, 3))
        (1, 2, 3)

        >>> into_tuple(5)
        ValueError: x should be tensor, list of tuple
    """
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("x should be tensor, list of tuple")


def into_orig_type(x, orig_type):
    """
        Inverts the transformation performed by the `into_tuple` function.

    This function takes an input `x` that is in tuple format and converts it back
    to its original type specified by `orig_type`. The conversion supports three
    types: `tuple`, `list`, and `torch.Tensor`.

    Args:
        x: A value in tuple format that needs to be converted back to its
           original type. It can be a tuple of tensors, a list of tensors,
           or a tensor itself.
        orig_type: The original type to which `x` should be converted. This
                   should be one of `tuple`, `list`, or `torch.Tensor`.

    Returns:
        The input `x` converted back to its original type specified by `orig_type`.

    Raises:
        ValueError: If `orig_type` is not one of the supported types
                    (`tuple`, `list`, or `torch.Tensor`).
        AssertionError: If `orig_type` is not handled properly.

    Examples:
        >>> into_orig_type((torch.tensor(1), torch.tensor(2)), tuple)
        (tensor(1), tensor(2))

        >>> into_orig_type((torch.tensor(1), torch.tensor(2)), list)
        [tensor(1), tensor(2)]

        >>> into_orig_type((torch.tensor(1),), torch.Tensor)
        tensor(1)
    """
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class ConcatAdaptLayer(nn.Module):
    """
        ConcatAdaptLayer is a PyTorch module that adapts input activations using
    enrollment embeddings by concatenating them. It is useful for scenarios
    where both normal and skip connections need adaptation.

    Attributes:
        ninputs (int): The number of input tensors to adapt.
        transform (nn.ModuleList): A list of linear transformation layers
            corresponding to the number of inputs.

    Args:
        indim (int): The dimensionality of the input activations.
        enrolldim (int): The dimensionality of the enrollment embeddings.
        ninputs (int, optional): The number of inputs to adapt. Defaults to 1.

    Methods:
        forward(main, enroll):
            Performs the forward pass of the layer.

    Returns:
        torch.Tensor or tuple or list: The adapted activations after applying
        the transformation.

    Raises:
        AssertionError: If the types of main and enroll do not match, or if the
        lengths of main and enroll do not match ninputs.

    Examples:
        >>> model = ConcatAdaptLayer(indim=128, enrolldim=64, ninputs=2)
        >>> main_input = (torch.randn(10, 128), torch.randn(10, 128))
        >>> enroll_input = (torch.randn(10, 64), torch.randn(10, 64))
        >>> output = model(main_input, enroll_input)
        >>> output[0].shape
        torch.Size([10, 128])
        >>> output[1].shape
        torch.Size([10, 128])
    """

    def __init__(self, indim, enrolldim, ninputs=1):
        super().__init__()
        self.ninputs = ninputs
        self.transform = nn.ModuleList(
            [nn.Linear(indim + enrolldim, indim) for _ in range(ninputs)]
        )

    def forward(self, main, enroll):
        """
        Initializes the ConcatAdaptLayer.

        Args:
            indim: int
                The input dimension for the main activations.
            enrolldim: int
                The dimension of the enrollment embeddings.
            ninputs: int, optional
                The number of input tensors (default is 1).

        Attributes:
            ninputs: int
                The number of input tensors.
            transform: nn.ModuleList
                A list of linear transformation layers for adapting
                the main activations with the enrollment embeddings.
        """
        assert type(main) is type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for transform, main0, enroll0 in zip(self.transform, main, enroll):
            out.append(
                transform(
                    torch.cat(
                        (main0, enroll0[:, :, None].expand(main0.shape)), dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
            )
        return into_orig_type(tuple(out), orig_type)


class MulAddAdaptLayer(nn.Module):
    """
    Layer that performs multiplication and addition adaptation.

    This layer adapts the main neural network activations by either
    multiplying or adding the enrollment embeddings, depending on the
    specified mode. The layer can be configured to perform addition
    or only multiplication based on the `do_addition` flag.

    Attributes:
        ninputs (int): The number of input tensors to be processed.
        do_addition (bool): A flag to indicate whether to perform
            addition along with multiplication. If True, the
            enrollment dimension must be twice the input dimension.

    Args:
        indim (int): The dimension of the input activations.
        enrolldim (int): The dimension of the enrollment embeddings.
        ninputs (int, optional): The number of input tensors (default is 1).
        do_addition (bool, optional): Flag to perform addition (default is True).

    Raises:
        AssertionError: If `do_addition` is True and `enrolldim` is not equal to
            twice `indim`, or if `do_addition` is False and `enrolldim` is
            not equal to `indim`.

    Examples:
        >>> layer = MulAddAdaptLayer(indim=128, enrolldim=256, ninputs=2)
        >>> main_input = torch.randn(10, 128)
        >>> enroll_input = torch.randn(10, 256)
        >>> output = layer(main_input, enroll_input)
        >>> print(output[0].shape)
        torch.Size([10, 128])

        >>> layer_no_add = MulAddAdaptLayer(indim=128, enrolldim=128,
        ...                                   ninputs=2, do_addition=False)
        >>> output_no_add = layer_no_add(main_input, enroll_input)
        >>> print(output_no_add[0].shape)
        torch.Size([10, 128])
    """

    def __init__(self, indim, enrolldim, ninputs=1, do_addition=True):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        if do_addition:
            assert enrolldim == 2 * indim, (enrolldim, indim)
        else:
            assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll):
        """
            A layer that performs multiplication and addition for adaptation.

        This layer adapts the activations of a neural network using an
        enrollment embedding. Depending on the configuration, it can either
        perform both multiplication and addition or just multiplication.

        Attributes:
            ninputs (int): The number of inputs to the layer.
            do_addition (bool): A flag to indicate if addition should be performed.
                If True, the enrollment dimension must be twice the input
                dimension. If False, they must be equal.

        Args:
            indim (int): The input dimension of the main activations.
            enrolldim (int): The dimension of the enrollment embeddings.
            ninputs (int, optional): The number of input tensors. Defaults to 1.
            do_addition (bool, optional): Whether to perform addition.
                Defaults to True.

        Raises:
            AssertionError: If the enrollment dimension does not match the
                expected dimension based on the configuration.

        Examples:
            >>> layer = MulAddAdaptLayer(indim=64, enrolldim=128, ninputs=2)
            >>> main_input = (torch.randn(10, 64), torch.randn(10, 64))
            >>> enroll_input = (torch.randn(10, 128), torch.randn(10, 128))
            >>> output = layer(main_input, enroll_input)
            >>> print(output[0].shape)  # Output shape should match (10, 64)

        Note:
            The input tensors can be provided as a tuple or list. This
            is useful for cases where both normal and skip connections
            are being adapted simultaneously.
            The forward method ensures that the main and enroll inputs
            are of the same type and length as specified by ninputs.
        """
        assert type(main) is type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs, (
            len(main),
            len(enroll),
            self.ninputs,
        )

        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(enroll0_mul[:, :, None] * main0 + enroll0_add[:, :, None])
            else:
                out.append(enroll0[:, :, None] * main0)
        return into_orig_type(tuple(out), orig_type)


# aliases for possible adaptation layer types
adaptation_layer_types = {
    "concat": ConcatAdaptLayer,
    "muladd": MulAddAdaptLayer,
    "mul": partial(MulAddAdaptLayer, do_addition=False),
}
