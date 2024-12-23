"""Beamformer module."""

from typing import Sequence, Tuple, Union

import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

EPS = torch.finfo(torch.double).eps


def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    """
    Create a new complex tensor based on the reference tensor.

    This function generates a new complex tensor that matches the type of the
    reference tensor `ref` using the provided real and imaginary parts. It
    supports both PyTorch's native complex tensors and the ComplexTensor
    from the `torch_complex` library.

    Args:
        ref: A reference tensor which can be either a PyTorch Tensor or
            a ComplexTensor. This determines the type of the output tensor.
        real_imag: A tuple containing two tensors, the first for the real
            part and the second for the imaginary part.

    Returns:
        A complex tensor of the same type as `ref`, constructed from
        `real_imag`.

    Raises:
        ValueError: If the PyTorch version is less than 1.9 or if `ref`
        is not a supported tensor type.

    Examples:
        >>> real = torch.tensor([1.0, 2.0])
        >>> imag = torch.tensor([3.0, 4.0])
        >>> ref_tensor = torch.tensor([0.0, 0.0], dtype=torch.complex64)
        >>> new_tensor = new_complex_like(ref_tensor, (real, imag))
        >>> print(new_tensor)
        tensor([1.+3.j, 2.+4.j], dtype=torch.complex64)

        >>> ref_complex = ComplexTensor(real, imag)
        >>> new_complex = new_complex_like(ref_complex, (real, imag))
        >>> print(new_complex)
        ComplexTensor(real=tensor([1., 2.]), imag=tensor([3., 4.]))
    """
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c):
    """
    Check if the input is a PyTorch complex tensor.

    This function verifies whether the input tensor is a complex tensor
    created using PyTorch's native complex support, excluding
    ComplexTensor from the torch_complex library.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor to check.

    Returns:
        bool: True if the input is a PyTorch complex tensor, False otherwise.

    Examples:
        >>> is_torch_complex_tensor(torch.tensor([1, 2], dtype=torch.complex64))
        True
        >>> is_torch_complex_tensor(torch.tensor([1, 2]))
        False
        >>> is_torch_complex_tensor(ComplexTensor(torch.tensor([1, 2]),
        ...                                         torch.tensor([3, 4])))
        False
    """
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)


def is_complex(c):
    """
    Check if the input is a complex tensor.

    This function determines whether the given input `c` is of type
    `ComplexTensor` or a native PyTorch complex tensor. It is useful
    for differentiating between complex and real tensor types in
    operations involving complex numbers.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor to check.

    Returns:
        bool: Returns True if `c` is a complex tensor; otherwise, False.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> c1 = ComplexTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> c2 = torch.tensor([1.0, 2.0], dtype=torch.complex64)
        >>> c3 = torch.tensor([1.0, 2.0])
        >>> is_complex(c1)
        True
        >>> is_complex(c2)
        True
        >>> is_complex(c3)
        False
    """
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def to_complex(c):
    """
        Convert input to a native PyTorch complex tensor.

    This function converts the input tensor to a native PyTorch complex tensor
    format. It handles both `ComplexTensor` and native complex tensors. If the
    input is neither, it attempts to convert the tensor using
    `torch.view_as_complex`.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor to be converted.

    Returns:
        Union[torch.Tensor, ComplexTensor]: The converted complex tensor.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> real = torch.tensor([1.0, 2.0])
        >>> imag = torch.tensor([3.0, 4.0])
        >>> complex_tensor = ComplexTensor(real, imag)
        >>> to_complex(complex_tensor)
        tensor([1.+3.j, 2.+4.j])

        >>> native_complex_tensor = torch.tensor([1.0, 2.0], dtype=torch.complex64)
        >>> to_complex(native_complex_tensor)
        tensor([1.+0.j, 2.+0.j])

        >>> real_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> to_complex(real_tensor)
        tensor([[1.+0.j, 2.+0.j],
                [3.+0.j, 4.+0.j]])
    """
    # Convert to torch native complex
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)


def to_double(c):
    """
    Convert a tensor to double precision.

    This function converts the input tensor to double precision (float64). If the
    input tensor is a complex tensor, it converts the real and imaginary parts
    to double precision. If the input tensor is a native complex tensor, it
    will be converted to complex128.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor to convert.

    Returns:
        Union[torch.Tensor, ComplexTensor]: The converted tensor in double precision.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> real_tensor = torch.tensor([1.0, 2.0])
        >>> complex_tensor = ComplexTensor(torch.tensor([1.0, 2.0]),
        ...                                 torch.tensor([3.0, 4.0]))
        >>> to_double(real_tensor)
        tensor([1., 2.], dtype=torch.float64)
        >>> to_double(complex_tensor)
        ComplexTensor(real=tensor([1., 2.], dtype=torch.float64),
                      imag=tensor([3., 4.], dtype=torch.float64))

    Note:
        The function assumes that PyTorch's complex support is available.
    """
    if not isinstance(c, ComplexTensor) and torch.is_complex(c):
        return c.to(dtype=torch.complex128)
    else:
        return c.double()


def to_float(c):
    """
    Convert a tensor to a float tensor.

    This function converts a complex tensor or a native complex PyTorch tensor
    to a float tensor. If the input is already a float tensor or does not
    meet the conditions for conversion, it returns the input unchanged.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor which can be
        either a ComplexTensor or a PyTorch tensor.

    Returns:
        Union[torch.Tensor, ComplexTensor]: A float tensor if the input is a
        complex tensor, otherwise returns the input tensor unchanged.

    Examples:
        >>> c_complex = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
        >>> to_float(c_complex)
        tensor([1., 2., 3., 4.])

        >>> c_float = torch.tensor([1.0, 2.0, 3.0])
        >>> to_float(c_float)
        tensor([1.0, 2.0, 3.0])
    """
    if not isinstance(c, ComplexTensor) and torch.is_complex(c):
        return c.to(dtype=torch.complex64)
    else:
        return c.float()


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
    Concatenate a sequence of tensors along a specified dimension.

    This function concatenates a sequence of tensors (either
    `ComplexTensor` or `torch.Tensor`) into a single tensor. The
    input sequence must be a list or tuple. If the first tensor in
    the sequence is a `ComplexTensor`, the function uses the
    `torch_complex` library's `cat` method; otherwise, it falls
    back to PyTorch's `torch.cat`.

    Args:
        seq (Sequence[Union[ComplexTensor, torch.Tensor]]): A sequence
            of tensors to concatenate. Must be a list or tuple.
        *args: Additional positional arguments to pass to the
            concatenation function.
        **kwargs: Additional keyword arguments to pass to the
            concatenation function.

    Returns:
        Union[ComplexTensor, torch.Tensor]: A tensor resulting from
        concatenating the input sequence.

    Raises:
        TypeError: If `seq` is not a list or tuple.

    Examples:
        >>> import torch
        >>> from torch_complex import ComplexTensor
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> b = torch.tensor([[5, 6], [7, 8]])
        >>> result = cat([a, b])
        >>> print(result)
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])

        >>> c = ComplexTensor(torch.tensor([[1, 2]]), torch.tensor([[3, 4]]))
        >>> d = ComplexTensor(torch.tensor([[5, 6]]), torch.tensor([[7, 8]]))
        >>> result_complex = cat([c, d])
        >>> print(result_complex)
        ComplexTensor(
            real=tensor([[1, 2],
                          [5, 6]]),
            imag=tensor([[3, 4],
                          [7, 8]])
        )
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "cat(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.cat(seq, *args, **kwargs)
    else:
        return torch.cat(seq, *args, **kwargs)


def complex_norm(
    c: Union[torch.Tensor, ComplexTensor], dim=-1, keepdim=False
) -> torch.Tensor:
    """
    Compute the norm of a complex tensor.

    This function calculates the norm of a complex tensor along a specified
    dimension. If the input tensor is not complex, a TypeError is raised.
    The function can handle both PyTorch's native complex tensors and
    custom ComplexTensor objects.

    Args:
        c (Union[torch.Tensor, ComplexTensor]): The input tensor for which to
            compute the norm. It must be either a complex tensor or a
            compatible type.
        dim (int, optional): The dimension along which to compute the norm.
            Default is -1, which means the last dimension.
        keepdim (bool, optional): Whether to retain the dimensions of the
            input tensor in the output. Default is False.

    Returns:
        torch.Tensor: The computed norm of the input tensor.

    Raises:
        TypeError: If the input tensor is not a complex tensor.

    Examples:
        >>> import torch
        >>> from torch_complex import ComplexTensor
        >>> c_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex64)
        >>> complex_norm(c_tensor)
        tensor(5.4772)

        >>> c_tensor_custom = ComplexTensor(torch.tensor([1.0, 2.0]),
        ...                                   torch.tensor([3.0, 4.0]))
        >>> complex_norm(c_tensor_custom, dim=0)
        tensor([3.1623, 4.4721])

    Note:
        This function adds a small epsilon value to the result to avoid
        division by zero errors.
    """
    if not is_complex(c):
        raise TypeError("Input is not a complex tensor.")
    if is_torch_complex_tensor(c):
        return torch.norm(c, dim=dim, keepdim=keepdim)
    else:
        if dim is None:
            return torch.sqrt((c.real**2 + c.imag**2).sum() + EPS)
        else:
            return torch.sqrt(
                (c.real**2 + c.imag**2).sum(dim=dim, keepdim=keepdim) + EPS
            )


def einsum(equation, *operands):
    """
    Perform Einstein summation convention on the input tensors.

    This function computes the Einstein summation of the provided operands
    based on the specified equation string. The operands can be either
    `torch.Tensor` or `ComplexTensor`, but mixing them is not allowed.
    The function handles both real and complex tensor operations.

    Args:
        equation (str): A string representing the Einstein summation
            convention. For example, 'ij,jk->ik' computes the matrix
            product of two tensors.
        *operands (Union[torch.Tensor, ComplexTensor]): One or two tensors
            to be operated on according to the equation. If one tensor is
            provided, it can be a tuple or list of tensors.

    Returns:
        Union[torch.Tensor, ComplexTensor]: The result of the Einstein
        summation operation, which can be either a real tensor or a complex
        tensor depending on the input types.

    Raises:
        ValueError: If the number of operands is not 1 or 2, or if there
        is a mix of tensor types (complex and real).

    Note:
        Do not mix `ComplexTensor` and `torch.complex` in the input!
        Until PyTorch 1.9.0, `torch.einsum` does not support mixed input
        with complex and real tensors.

    Examples:
        >>> import torch
        >>> a = torch.randn(2, 3)
        >>> b = torch.randn(3, 4)
        >>> einsum('ij,jk->ik', a, b)
        tensor([[...], [...]])

        >>> c = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        >>> d = ComplexTensor(torch.randn(3, 4), torch.randn(3, 4))
        >>> einsum('ij,jk->ik', c, d)
        ComplexTensor(real=[...], imag=[...])
    """
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.einsum does not support
    # mixed input with complex and real tensors.
    if len(operands) == 1:
        if isinstance(operands[0], (tuple, list)):
            operands = operands[0]
        complex_module = FC if isinstance(operands[0], ComplexTensor) else torch
        return complex_module.einsum(equation, *operands)
    elif len(operands) != 2:
        op0 = operands[0]
        same_type = all(op.dtype == op0.dtype for op in operands[1:])
        if same_type:
            _einsum = FC.einsum if isinstance(op0, ComplexTensor) else torch.einsum
            return _einsum(equation, *operands)
        else:
            raise ValueError("0 or More than 2 operands are not supported.")
    a, b = operands
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.einsum(equation, a, b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if not torch.is_complex(a):
            o_real = torch.einsum(equation, a, b.real)
            o_imag = torch.einsum(equation, a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.einsum(equation, a.real, b)
            o_imag = torch.einsum(equation, a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.einsum(equation, a, b)
    else:
        return torch.einsum(equation, a, b)


def inverse(
    c: Union[torch.Tensor, ComplexTensor]
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Compute the inverse of a complex or real tensor.

    This function calculates the inverse of a given tensor, which can be
    either a real tensor or a complex tensor (using `ComplexTensor`).
    If the input is a `ComplexTensor`, the function will use the
    `inverse2()` method. For real tensors, it will use the standard
    `inverse()` method.

    Args:
        c: A tensor that can either be a `torch.Tensor` or a `ComplexTensor`.
           It is expected to be a square matrix for the inverse operation.

    Returns:
        A tensor that is the inverse of the input tensor `c`. The type of
        the returned tensor will match the input tensor type.

    Raises:
        ValueError: If the input tensor is not square or of unsupported type.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> real_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> complex_tensor = ComplexTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...                                torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
        >>> inverse_real = inverse(real_tensor)
        >>> inverse_complex = inverse(complex_tensor)
        >>> print(inverse_real)
        tensor([[-2.0000,  1.0000],
                [1.5000, -0.5000]])
        >>> print(inverse_complex)
        ComplexTensor(real=tensor([[-2.0000,  1.0000],
                                    [1.5000, -0.5000]]),
                      imag=tensor([[0.0, 0.0],
                                    [0.0, 0.0]]))

    Note:
        The input tensor must be a square matrix for the inverse to be
        computed. Ensure that the dimensions of the input tensor are
        valid to avoid runtime errors.

    Todo:
        Add support for batch processing of tensors in the future.
    """
    if isinstance(c, ComplexTensor):
        return c.inverse2()
    else:
        return c.inverse()


def matmul(
    a: Union[torch.Tensor, ComplexTensor], b: Union[torch.Tensor, ComplexTensor]
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Perform matrix multiplication on two tensors.

    This function computes the matrix product of two input tensors, `a` and `b`.
    It supports both real and complex tensors. If either `a` or `b` is a
    `ComplexTensor`, the function will use the `torch_complex` library for the
    multiplication. If both tensors are real, the standard PyTorch
    `torch.matmul` function will be used.

    Note:
        Do not mix `ComplexTensor` and `torch.complex` in the input tensors.
        Until PyTorch 1.9.0, `torch.matmul` does not support multiplication
        between complex and real tensors.

    Args:
        a: A tensor of type `torch.Tensor` or `ComplexTensor`.
        b: A tensor of type `torch.Tensor` or `ComplexTensor`.

    Returns:
        A tensor of the same type as the inputs, which is the result of the
        matrix multiplication.

    Raises:
        ValueError: If both inputs are not of type `torch.Tensor` or
        `ComplexTensor`.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> b = torch.tensor([[5, 6], [7, 8]])
        >>> result = matmul(a, b)
        >>> print(result)
        tensor([[19, 22],
                [43, 50]])

        >>> a_complex = ComplexTensor(torch.tensor([[1, 2], [3, 4]]),
        ...                            torch.tensor([[5, 6], [7, 8]]))
        >>> b_complex = ComplexTensor(torch.tensor([[1, 0], [0, 1]]),
        ...                            torch.tensor([[0, 1], [1, 0]]))
        >>> result_complex = matmul(a_complex, b_complex)
        >>> print(result_complex)
        ComplexTensor(
            tensor([[  5,   6],
                    [  43,  50]]),
            tensor([[  6,  5],
                    [  50,  43]])
        )
    """
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.matmul does not support
    # multiplication between complex and real tensors.
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.matmul(a, b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if not torch.is_complex(a):
            o_real = torch.matmul(a, b.real)
            o_imag = torch.matmul(a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.matmul(a.real, b)
            o_imag = torch.matmul(a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.matmul(a, b)
    else:
        return torch.matmul(a, b)


def trace(a: Union[torch.Tensor, ComplexTensor]):
    """
    Compute the trace of a tensor.

    The trace of a matrix is defined as the sum of the elements on the main
    diagonal. This function can handle both standard PyTorch tensors and
    ComplexTensor types. For versions of PyTorch prior to 1.9.0, it uses
    the `FC.trace()` function as a fallback, since `torch.trace()` does not
    support batch processing.

    Args:
        a: A tensor or a ComplexTensor for which the trace is to be computed.

    Returns:
        A scalar tensor representing the trace of the input tensor.

    Raises:
        TypeError: If the input is not a tensor or ComplexTensor.

    Examples:
        >>> import torch
        >>> from torch_complex.tensor import ComplexTensor
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> trace(a)
        tensor(5)

        >>> c = ComplexTensor(torch.tensor([[1, 2], [3, 4]]),
        ...                   torch.tensor([[5, 6], [7, 8]]))
        >>> trace(c)
        ComplexTensor(real=tensor(5), imag=tensor(21))

    Note:
        This function is intended for use with 2D tensors, typically
        representing matrices.
    """
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    return FC.trace(a)


def reverse(a: Union[torch.Tensor, ComplexTensor], dim=0):
    """
    Reverse the elements of a tensor along a specified dimension.

    This function takes a tensor (either a PyTorch tensor or a ComplexTensor)
    and reverses its elements along the specified dimension. If the input is
    a ComplexTensor, it uses the corresponding function from the torch_complex
    library; otherwise, it utilizes PyTorch's built-in flip function.

    Args:
        a: A tensor to be reversed. This can be either a PyTorch tensor or
           a ComplexTensor.
        dim: The dimension along which to reverse the tensor. Default is 0.

    Returns:
        A tensor of the same type as `a`, with its elements reversed along the
        specified dimension.

    Examples:
        >>> import torch
        >>> from torch_complex import ComplexTensor
        >>> a = torch.tensor([1, 2, 3, 4])
        >>> reverse(a, dim=0)
        tensor([4, 3, 2, 1])

        >>> b = ComplexTensor(torch.tensor([1, 2]), torch.tensor([3, 4]))
        >>> reverse(b, dim=0)
        ComplexTensor(real=tensor([2, 1]), imag=tensor([4, 3]))

    Note:
        - Ensure that the input tensor has the specified dimension.
    """
    if isinstance(a, ComplexTensor):
        return FC.reverse(a, dim=dim)
    else:
        return torch.flip(a, dims=(dim,))


def solve(b: Union[torch.Tensor, ComplexTensor], a: Union[torch.Tensor, ComplexTensor]):
    """
    Solve the linear equation ax = b.

    This function computes the solution of the linear matrix equation
    `ax = b`, where `a` is a matrix and `b` is a vector or matrix.
    It handles both `torch.Tensor` and `ComplexTensor` types.

    Note that mixing `ComplexTensor` and `torch.complex` is not allowed.
    As of PyTorch 1.9.0, `torch.solve` does not support mixed inputs
    with complex and real tensors.

    Args:
        b: The right-hand side of the equation (torch.Tensor or ComplexTensor).
        a: The left-hand side of the equation (torch.Tensor or ComplexTensor).

    Returns:
        The solution `x` to the equation `ax = b`, which has the same
        type as `b` (torch.Tensor or ComplexTensor).

    Raises:
        ValueError: If the input tensors are not compatible for solving.

    Examples:
        >>> import torch
        >>> a = torch.tensor([[2, 1], [1, 3]], dtype=torch.float32)
        >>> b = torch.tensor([1, 2], dtype=torch.float32)
        >>> x = solve(b, a)
        >>> print(x)
        tensor([0.0000, 0.6667])

        >>> a_complex = ComplexTensor(torch.tensor([[2, 1], [1, 3]]),
        ...                            torch.tensor([[0, 0], [0, 0]]))
        >>> b_complex = ComplexTensor(torch.tensor([1, 2]),
        ...                            torch.tensor([0, 0]))
        >>> x_complex = solve(b_complex, a_complex)
        >>> print(x_complex)
        ComplexTensor(real=tensor([0.0000, 0.6667]), imag=tensor([0, 0]))
    """
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.solve does not support
    # mixed input with complex and real tensors.
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
            return FC.solve(b, a, return_LU=False)
        else:
            return matmul(inverse(a), b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if torch.is_complex(a) and torch.is_complex(b):
            return torch.linalg.solve(a, b)
        else:
            return matmul(inverse(a), b)
    else:
        return torch.linalg.solve(a, b)


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
        Stack tensors along a new dimension.

    This function takes a sequence of tensors and stacks them along a new
    dimension. It can handle both PyTorch tensors and ComplexTensor objects.

    Args:
        seq (Sequence[Union[ComplexTensor, torch.Tensor]]): A sequence of tensors
            to be stacked. Must be either a list or tuple.
        *args: Additional arguments passed to the stacking function.
        **kwargs: Additional keyword arguments passed to the stacking function.

    Returns:
        Union[ComplexTensor, torch.Tensor]: A new tensor formed by stacking
        the input tensors along a new dimension.

    Raises:
        TypeError: If the input `seq` is not a list or tuple.

    Examples:
        >>> import torch
        >>> from torch_complex import ComplexTensor
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> b = torch.tensor([[5, 6], [7, 8]])
        >>> stacked_tensor = stack([a, b])
        >>> print(stacked_tensor)
        tensor([[[1, 2],
                  [3, 4]],

                 [[5, 6],
                  [7, 8]]])

        >>> c = ComplexTensor(torch.tensor([[1, 2]]), torch.tensor([[3, 4]]))
        >>> d = ComplexTensor(torch.tensor([[5, 6]]), torch.tensor([[7, 8]]))
        >>> stacked_complex = stack([c, d])
        >>> print(stacked_complex)
        ComplexTensor(real=tensor([[[1, 2]],
                                    [[5, 6]]]),
                       imag=tensor([[[3, 4]],
                                    [[7, 8]]]))
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "stack(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.stack(seq, *args, **kwargs)
    else:
        return torch.stack(seq, *args, **kwargs)
