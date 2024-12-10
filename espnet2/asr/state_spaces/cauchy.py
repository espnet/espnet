# This code is derived from https://github.com/HazyResearch/state-spaces

import torch
from cauchy_mult import (
    cauchy_mult_bwd,
    cauchy_mult_fwd,
    cauchy_mult_sym_bwd,
    cauchy_mult_sym_fwd,
)
from einops import rearrange


def cauchy_mult_torch(
    v: torch.Tensor, z: torch.Tensor, w: torch.Tensor, symmetric=True
) -> torch.Tensor:
    """
    Compute the Cauchy kernel multiplication for tensors.

    This function computes the Cauchy kernel defined by the input tensors
    `v`, `z`, and `w`. The computation can be done under the assumption that 
    `v` and `w` are symmetric (i.e., they contain complex conjugate pairs).

    Args:
        v (torch.Tensor): Input tensor of shape (B, N) representing the first set 
            of data points.
        z (torch.Tensor): Input tensor of shape (L) representing the second set 
            of data points.
        w (torch.Tensor): Input tensor of shape (B, N) representing the third 
            set of data points.
        symmetric (bool, optional): A flag indicating whether `v` and `w` are 
            assumed to contain complex conjugate pairs. Defaults to True.

    Returns:
        torch.Tensor: The result of the Cauchy kernel multiplication with shape 
        (B, L).

    Raises:
        AssertionError: If `symmetric` is True and the last dimension of `v` 
        is not even.

    Examples:
        >>> v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> z = torch.tensor([1.0, 2.0, 3.0])
        >>> w = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> result = cauchy_mult_torch(v, z, w, symmetric=True)
        >>> print(result)
        tensor([...])  # Expected output based on the Cauchy kernel computation.

    Note:
        The function utilizes broadcasting to align the shapes of `v`, `z`, 
        and `w` for computation. Ensure that `v` and `w` have the same last 
        dimension size before calling this function.

    Todo:
        - Extend functionality to handle cases where `symmetric` is False 
          and add tests for various input shapes.
    """
    if not symmetric:
        return (
            rearrange(v, "b n -> b 1 n")
            / (rearrange(z, "l -> l 1") - rearrange(w, "b n -> b 1 n"))
        ).sum(dim=-1)
    else:
        N = v.shape[-1]
        assert N % 2 == 0
        vv = rearrange(v[:, : N // 2], "b n -> b 1 n")
        zz = rearrange(z, "l -> l 1")
        ww = rearrange(w[:, : N // 2], "b n -> b 1 n")
        return 2 * (
            (zz * vv.real - vv.real * ww.real - vv.imag * ww.imag)
            / (zz * zz - 2 * zz * ww.real + ww.abs().square())
        ).sum(dim=-1)


def cauchy_mult_keops(v, z, w):
    """
    Compute the Cauchy product of tensors using Keops for efficient computation.

    This function utilizes the Keops library to perform Cauchy multiplication 
    on tensors `v`, `z`, and `w`. The operation can be particularly useful in 
    applications where the Cauchy kernel is needed, such as in machine learning 
    and signal processing.

    Args:
        v (torch.Tensor): A tensor of shape (B, N), where B is the batch size 
            and N is the number of features.
        z (torch.Tensor): A tensor of shape (L), where L is the length of the 
            evaluation points.
        w (torch.Tensor): A tensor of shape (B, N), similar to `v`, which 
            serves as another set of features for the Cauchy multiplication.

    Returns:
        torch.Tensor: A tensor of shape (B, L) representing the result of the 
        Cauchy multiplication.

    Raises:
        NotImplementedError: If the shapes of the input tensors do not comply 
        with the expected sizes or if CUDA tensors are not used.

    Examples:
        >>> v = torch.randn(2, 4)  # Batch size of 2, 4 features
        >>> z = torch.randn(3)     # 3 evaluation points
        >>> w = torch.randn(2, 4)  # Same shape as v
        >>> result = cauchy_mult_keops(v, z, w)
        >>> print(result.shape)    # Output: torch.Size([2, 3])

    Note:
        This function requires the PyKeops library to be installed and properly 
        configured for CUDA support.

    Todo:
        - Add support for additional input shapes if necessary.
    """
    from pykeops.torch import LazyTensor

    v_l = LazyTensor(rearrange(v, "b N -> b 1 N 1"))
    z_l = LazyTensor(rearrange(z, "L -> 1 L 1 1"))
    w_l = LazyTensor(rearrange(w, "b N -> b 1 N 1"))
    sub = z_l - w_l  # (b N L 1), for some reason it doesn't display the last dimension
    div = v_l / sub
    s = div.sum(dim=2, backend="GPU")
    return s.squeeze(-1)


def _cauchy_mult(v, z, w, symmetric=True):
    if not symmetric:
        return CauchyMultiply.apply(v, z, w)
    else:
        return CauchyMultiplySymmetric.apply(v, z, w)


def cauchy_mult(v, z, w, symmetric=True):
    """
    Compute the Cauchy multiplication of input tensors.

    This function performs Cauchy multiplication of input tensors `v`, `z`, and `w`.
    It can handle both symmetric and non-symmetric cases, depending on the 
    value of the `symmetric` parameter. In the symmetric case, it assumes that 
    the input tensors contain complex conjugate pairs.

    Attributes:
        v (torch.Tensor): A tensor of shape (B, N) where B is the batch size 
                        and N is the size of the input.
        z (torch.Tensor): A tensor of shape (L) where L is the length of the 
                        second input.
        w (torch.Tensor): A tensor of shape (B, N) similar to `v`.
        symmetric (bool): A flag indicating whether to treat the inputs as 
                        symmetric. Defaults to True.

    Args:
        v (torch.Tensor): The first input tensor.
        z (torch.Tensor): The second input tensor.
        w (torch.Tensor): The third input tensor.
        symmetric (bool): Indicates if `v` and `w` are symmetric. Default is True.

    Returns:
        torch.Tensor: The result of the Cauchy multiplication of the input tensors.

    Raises:
        AssertionError: If the shapes of the input tensors are not compatible.
        NotImplementedError: If the input tensor sizes do not meet specific criteria.

    Examples:
        >>> v = torch.randn(2, 4)  # 2 batches, 4 elements
        >>> z = torch.randn(3)      # 3 elements
        >>> w = torch.randn(2, 4)   # 2 batches, 4 elements
        >>> result = cauchy_mult(v, z, w, symmetric=True)
        >>> print(result.shape)     # Should print torch.Size([2, 3])

    Note:
        The function assumes that the last dimension of `v` and `w` are equal.
    """
    v, w = torch.broadcast_tensors(v, w)
    shape = v.shape
    # z_shape = z.shape
    z = z.squeeze()
    assert len(z.shape) == 1

    v = v.contiguous()
    w = w.contiguous()
    z = z.contiguous()

    N = v.size(-1)
    assert w.size(-1) == N
    y = _cauchy_mult(v.view(-1, N), z, w.view(-1, N), symmetric=symmetric)
    y = y.view(*shape[:-1], z.size(-1))
    # y = z.new_zeros(*shape[:-1], z.size(-1))
    return y


class CauchyMultiply(torch.autograd.Function):
    """
    CauchyMultiply is a PyTorch autograd function that performs the forward and 
    backward passes for the Cauchy multiplication operation, which is a key 
    operation in various applications, including signal processing and 
    machine learning.

    This function is designed to operate on CUDA tensors and supports only 
    specific input dimensions for efficiency. It computes the multiplication 
    using a Cauchy kernel defined by the input tensors.

    Attributes:
        v (torch.Tensor): A tensor of shape (B, N) where B is the batch size and 
                        N is the dimension of the input vectors.
        z (torch.Tensor): A tensor of shape (L) which represents the points 
                        at which the Cauchy kernel is evaluated.
        w (torch.Tensor): A tensor of shape (B, N) similar to v, used in the 
                        Cauchy multiplication.
        
    Args:
        ctx: The context object used to save tensors for backward computation.
        v (torch.Tensor): Input tensor of shape (B, N).
        z (torch.Tensor): Input tensor of shape (L).
        w (torch.Tensor): Input tensor of shape (B, N).

    Returns:
        torch.Tensor: The result of the Cauchy multiplication, a tensor of shape 
                    (B, L).

    Raises:
        NotImplementedError: If the input tensor sizes do not meet the 
                            requirements (N must be a power of 2, L must be 
                            a multiple of 32, and all tensors must be on CUDA).

    Examples:
        # Example usage of CauchyMultiply in a forward pass
        v = torch.randn(2, 64, device='cuda')  # Batch of 2, dimension 64
        z = torch.randn(128, device='cuda')     # Points at which to evaluate
        w = torch.randn(2, 64, device='cuda')   # Another tensor of the same size as v
        result = CauchyMultiply.apply(v, z, w)
        
        # Example usage in a backward pass
        dout = torch.randn_like(result)          # Gradient of the output
        dv, _, dw = CauchyMultiply.backward(ctx, dout)

    Note:
        This class is intended for advanced users who are familiar with 
        PyTorch's autograd functionality and CUDA programming.

    Todo:
        - Extend support for additional tensor shapes and dimensions.
    """
    @staticmethod
    def forward(ctx, v, z, w):
        """
        Wrap the CUDA method to compute Cauchy multiplication with shape handling.

        This function computes the Cauchy multiplication of input tensors `v`, `z`, 
        and `w`. It supports both symmetric and non-symmetric configurations. The 
        inputs are broadcasted to ensure compatible shapes, and the resulting tensor 
        is reshaped accordingly.

        Args:
            v (torch.Tensor): A tensor of shape (B, N) where B is the batch size 
                and N is the number of elements in each input vector.
            z (torch.Tensor): A tensor of shape (L) where L is the number of 
                evaluation points.
            w (torch.Tensor): A tensor of shape (B, N) similar to `v`.
            symmetric (bool, optional): If True, assumes that `v` and `w` contain 
                complex conjugate pairs. Defaults to True.

        Returns:
            torch.Tensor: A tensor of shape (B, L) containing the result of the 
            Cauchy multiplication.

        Raises:
            AssertionError: If the shapes of `v`, `w`, or `z` are not compatible 
            or if `z` does not have the expected shape.
            NotImplementedError: If `N` or `L` do not meet the specified criteria 
            for supported sizes.

        Examples:
            >>> v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> z = torch.tensor([0.5, 1.0, 1.5])
            >>> w = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
            >>> result = cauchy_mult(v, z, w)
            >>> print(result.shape)  # Output: torch.Size([2, 3])

            >>> result_symmetric = cauchy_mult(v, z, w, symmetric=True)
            >>> print(result_symmetric.shape)  # Output: torch.Size([2, 3])
        """
        batch, N = v.shape
        # supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        supported_N_values = [1 << log_n for log_n in [6]]
        L = z.shape[-1]
        if N not in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        if L % 32 != 0:
            raise NotImplementedError("Only support L values that are multiples of 32")
        if not v.is_cuda and z.is_cuda and w.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(v, z, w)
        return cauchy_mult_fwd(v, z, w)

    @staticmethod
    def backward(ctx, dout):
        """
        Computes the gradient of the Cauchy multiplication operation.

        This method is part of the CauchyMultiply autograd function. It takes the 
        saved tensors from the forward pass and computes the gradients of the input 
        tensors `v` and `w` based on the output gradient `dout`.

        Args:
            ctx: The context object that stores information from the forward pass.
            dout (torch.Tensor): The gradient of the output from the forward pass.

        Returns:
            tuple: A tuple containing the gradients with respect to the inputs:
                - dv (torch.Tensor): The gradient of the input tensor `v`.
                - None: No gradient for the `z` tensor, as it is not trainable.
                - dw (torch.Tensor): The gradient of the input tensor `w`.

        Raises:
            NotImplementedError: If the operation is not supported for the given input 
            sizes or types.

        Examples:
            >>> import torch
            >>> v = torch.randn(2, 64, device='cuda', requires_grad=True)
            >>> z = torch.randn(32, device='cuda', requires_grad=False)
            >>> w = torch.randn(2, 64, device='cuda', requires_grad=True)
            >>> output = CauchyMultiply.apply(v, z, w)
            >>> output.sum().backward()  # Compute gradients
            >>> v.grad  # Access gradient of v
        >>> w.grad  # Access gradient of w
        """
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_bwd(v, z, w, dout)
        return dv, None, dw


class CauchyMultiplySymmetric(torch.autograd.Function):
    """
    Perform Cauchy multiplication for symmetric tensors.

    This class implements the forward and backward methods for computing 
    the Cauchy multiplication of input tensors while leveraging PyTorch's 
    autograd functionality. The multiplication assumes that the input 
    tensors exhibit symmetry, which allows for optimized calculations.

    Attributes:
        None

    Args:
        v (torch.Tensor): A tensor of shape (B, N) representing the first input.
        z (torch.Tensor): A tensor of shape (L) representing the second input.
        w (torch.Tensor): A tensor of shape (B, N) representing the third input.

    Returns:
        torch.Tensor: The result of the Cauchy multiplication.

    Raises:
        NotImplementedError: If N is not in supported values or if L exceeds 
        the maximum limit or if input tensors are not CUDA compatible.

    Examples:
        >>> v = torch.randn(2, 64, device='cuda')
        >>> z = torch.randn(32, device='cuda')
        >>> w = torch.randn(2, 64, device='cuda')
        >>> result = CauchyMultiplySymmetric.apply(v, z, w)
        >>> print(result.shape)  # Should output: (2, 32)

    Note:
        The function currently supports N values that are powers of two 
        up to 1024. L must be a multiple of 32 and cannot exceed a 
        predefined maximum value.

    Todo:
        - Expand support for additional N values and L sizes in future 
        implementations.
    """
    @staticmethod
    def forward(ctx, v, z, w):
        """
        Implements the forward and backward passes for symmetric Cauchy 
        multiplication.

        This class leverages PyTorch's autograd functionality to compute the 
        Cauchy product of tensors in a symmetric manner. The forward method 
        computes the result based on the input tensors, while the backward 
        method calculates the gradients for backpropagation.

        Attributes:
            None

        Args:
            ctx: The context object that can be used to stash information 
                for backward computation.
            v (torch.Tensor): Input tensor of shape (B, N), where B is the 
                batch size and N is the dimension of the input.
            z (torch.Tensor): Input tensor of shape (L), where L is the 
                dimension of the second input.
            w (torch.Tensor): Input tensor of shape (B, N), similar to v.

        Returns:
            torch.Tensor: The result of the Cauchy multiplication of v, z, 
            and w.

        Raises:
            NotImplementedError: If the input sizes do not meet the expected 
            conditions:
                - N must be a power of two (supported values are [1 << log_n 
                for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).
                - L must be a multiple of 32.
                - All tensors must be on CUDA if v is not a CUDA tensor.

        Examples:
            To compute the symmetric Cauchy product:
            
            ```python
            import torch

            v = torch.randn(32, 64).cuda()  # Batch of 32, dimension 64
            z = torch.randn(128).cuda()      # Dimension 128
            w = torch.randn(32, 64).cuda()   # Batch of 32, dimension 64

            result = CauchyMultiplySymmetric.apply(v, z, w)
            ```

        Note:
            Ensure that the input tensors are properly shaped and on the 
            correct device (CUDA) before calling the forward method.

        Todo:
            Extend support for larger input sizes or additional functionalities.
        """
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        L = z.shape[-1]
        if N not in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError("Only support L values <= {max_L_value}")
        if not v.is_cuda and z.is_cuda and w.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(v, z, w)
        return cauchy_mult_sym_fwd(v, z, w)

    @staticmethod
    def backward(ctx, dout):
        """
        Compute the gradients of the Cauchy multiplication operation.

        This method is called during the backward pass of the CauchyMultiply
        and CauchyMultiplySymmetric operations. It takes the saved tensors
        from the forward pass and computes the gradients with respect to the
        input tensors.

        Args:
            ctx: The context object that contains saved tensors from the forward
                pass.
            dout (torch.Tensor): The gradients of the output with respect to some
                loss, which will be propagated back to the input tensors.

        Returns:
            tuple: A tuple containing:
                - dv (torch.Tensor): The gradient of the loss with respect to the
                input tensor `v`.
                - None: Placeholder for the gradient with respect to `z`, which is
                not required.
                - dw (torch.Tensor): The gradient of the loss with respect to the
                input tensor `w`.

        Examples:
            >>> v = torch.randn(10, 64, device='cuda')
            >>> z = torch.randn(128, device='cuda')
            >>> w = torch.randn(10, 64, device='cuda')
            >>> cauchy_mul = CauchyMultiply.apply(v, z, w)
            >>> cauchy_mul.backward(torch.ones_like(cauchy_mul))
        
        Note:
            The backward function assumes that the input tensors are CUDA tensors
            and raises an error if they are not.

        Raises:
            NotImplementedError: If the input tensor sizes do not meet the
            specified conditions.
        """
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_sym_bwd(v, z, w, dout)
        return dv, None, dw
