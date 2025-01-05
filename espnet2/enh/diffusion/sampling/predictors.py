# Adapted from https://github.com/yang-song/score_sde_pytorch/
# and https://github.com/sp-uhh/sgmse
import abc

import numpy as np
import torch


class Predictor(abc.ABC):
    """
    The abstract class for a predictor algorithm.

    This class serves as a base for various predictor algorithms that can be
    implemented for different types of stochastic differential equations (SDEs).
    The predictor utilizes a score function and the SDE model to update its
    predictions over time.

    Attributes:
        sde: The stochastic differential equation model.
        rsde: The reverse stochastic differential equation derived from the
            score function.
        score_fn: The function used to compute the score.
        probability_flow: A boolean indicating if the probability flow is
            utilized.

    Args:
        sde: The SDE model to be used.
        score_fn: The score function that will be used for the predictions.
        probability_flow: (Optional) A boolean indicating whether to use
            probability flow (default: False).

    Methods:
        update_fn(x, t, *args): An abstract method that updates the predictor
            state based on the current state and time.
        debug_update_fn(x, t, *args): Raises a NotImplementedError indicating
            that the debug update function is not implemented.

    Raises:
        NotImplementedError: If the debug_update_fn is called without
            implementation.

    Examples:
        >>> predictor = SomeConcretePredictor(sde, score_fn)
        >>> next_state, mean_state = predictor.update_fn(current_state, time)
    """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """
            One update of the predictor.

        This method computes one update step for the predictor algorithm.
        It takes the current state and time step as input and returns the
        updated state along with the mean state, which is useful for
        denoising.

        Args:
            x (torch.Tensor): A PyTorch tensor representing the current state.
            t (torch.Tensor): A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): A PyTorch tensor of the next state.
                - x_mean (torch.Tensor): A PyTorch tensor representing the next state
                  without random noise, useful for denoising.

        Examples:
            # Example usage:
            x_current = torch.tensor([[0.0], [1.0]])
            t_current = torch.tensor([0.1])
            next_state, next_state_mean = predictor.update_fn(x_current, t_current)

        Note:
            This method is abstract and must be implemented in subclasses of the
            Predictor class.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        """
            Debug update function for the Predictor class.

        This function is intended for debugging purposes and should be
        implemented in subclasses of Predictor to provide specific
        functionality. Currently, it raises a NotImplementedError to
        indicate that the debug update function has not been defined
        for the specific predictor instance.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments that may be used in
                specific implementations.

        Raises:
            NotImplementedError: If the debug update function is not
                implemented for the predictor.

        Examples:
            # Example usage (will raise NotImplementedError)
            predictor = SomePredictorClass(...)
            x, t = torch.randn(1, 3), torch.tensor(0.0)
            predictor.debug_update_fn(x, t)

        Note:
            Subclasses should override this method to provide a
            meaningful implementation for debugging.
        """
        raise NotImplementedError(
            f"Debug update function not implemented for predictor {self}."
        )


class EulerMaruyamaPredictor(Predictor):
    """
    Euler-Maruyama predictor for stochastic differential equations.

    This class implements the Euler-Maruyama method, which is a numerical
    method for simulating stochastic differential equations (SDEs). It
    predicts the next state based on the current state and the given
    parameters of the SDE.

    Args:
        sde: The stochastic differential equation to be solved.
        score_fn: The score function used to evaluate the gradient.
        probability_flow (bool): If True, use probability flow; defaults to False.

    Attributes:
        sde: The stochastic differential equation.
        rsde: The reverse SDE corresponding to the forward SDE.
        score_fn: The score function.
        probability_flow: Boolean indicating the use of probability flow.

    Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor representing the next state without
                 random noise, useful for denoising.

    Examples:
        >>> predictor = EulerMaruyamaPredictor(sde, score_fn)
        >>> x_next, x_mean = predictor.update_fn(x_current, t_current)

    Note:
        This predictor is typically used in the context of sampling
        from a diffusion process.

    Raises:
        NotImplementedError: If the update function is not implemented.
    """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        f, g = self.rsde.sde(x, t, *args)
        x_mean = x + f * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    """
    Predictor for reverse diffusion sampling.

    This class implements the reverse diffusion process using a score-based
    generative model. It is designed to perform sampling by reversing the
    diffusion process through learned score functions.

    Attributes:
        sde: The stochastic differential equation object used for sampling.
        score_fn: The score function used to guide the reverse diffusion.
        probability_flow: A boolean indicating whether to use probability flow
            or not. If True, it applies a deterministic approach.

    Args:
        sde: An instance of a stochastic differential equation.
        score_fn: A callable that estimates the score.
        probability_flow (bool): A flag to indicate the use of probability
            flow (default is False).

    Returns:
        x: A PyTorch tensor representing the next state in the diffusion
            process.
        x_mean: A PyTorch tensor representing the mean of the next state
            without added noise, useful for denoising.

    Examples:
        >>> predictor = ReverseDiffusionPredictor(sde, score_fn)
        >>> x, x_mean = predictor.update_fn(current_state, current_time)

    Note:
        This predictor assumes that the input `x` and `t` are properly
        formatted tensors, and additional arguments can be passed as needed.

    Raises:
        NotImplementedError: If the update function is called without
            implementing the necessary functionality.
    """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        """
            Predictor for reverse diffusion processes.

        This class implements the update function for reverse diffusion,
        utilizing a specified stochastic differential equation (SDE) and
        score function. The reverse diffusion process aims to recover
        the original data from a noisy version through a learned score
        function.

        Attributes:
            sde: A stochastic differential equation object.
            rsde: A reverse stochastic differential equation object.
            score_fn: A function that computes the score (gradient of log
                probability).
            probability_flow: A boolean indicating whether to use probability
                flow in the update.

        Args:
            sde: An instance of the SDE to be used.
            score_fn: A callable that returns the score function.
            probability_flow: A boolean flag for enabling probability flow
                (default: False).

        Returns:
            x: A PyTorch tensor representing the next state after the update.
            x_mean: A PyTorch tensor representing the next state without
                random noise, useful for denoising.

        Examples:
            >>> predictor = ReverseDiffusionPredictor(sde, score_fn)
            >>> x_next, x_mean = predictor.update_fn(x_current, t_current)

        Note:
            This implementation assumes that the input tensor `x` is
            compatible with the dimensions expected by the SDE and score
            function.

        Raises:
            ValueError: If the input tensor `x` or time step `t` are not
                of the expected shape or type.
        """
        f, g = self.rsde.discretize(x, t, *args)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    """
    An empty predictor that does nothing.

    This class serves as a placeholder for situations where no prediction
    is required. It inherits from the `Predictor` abstract base class and
    implements the `update_fn` method to simply return the input state
    without any modifications.

    Attributes:
        None

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A tuple containing:
            - x: A PyTorch tensor representing the current state (unchanged).
            - x: A PyTorch tensor representing the next state (unchanged).

    Examples:
        >>> predictor = NonePredictor()
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t = torch.tensor(0)
        >>> next_state, _ = predictor.update_fn(x, t)
        >>> print(next_state)
        tensor([[1.0, 2.0],
                [3.0, 4.0]])
    """

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args):
        """
            One update of the predictor.

        This method is responsible for computing the next state of the predictor
        based on the current state and time step. The behavior of this method is
        determined by the specific implementation of the predictor class.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes.

        Returns:
            A tuple containing:
                x: A PyTorch tensor of the next state.
                x_mean: A PyTorch tensor representing the next state without random
                        noise, useful for denoising.

        Examples:
            # Example usage:
            x_current = torch.tensor([[0.0, 0.0]])
            t_current = torch.tensor([1.0])
            next_state, next_mean = predictor.update_fn(x_current, t_current)

        Note:
            The actual behavior of this method will depend on the specific
            implementation of the predictor class (e.g., EulerMaruyamaPredictor
            or ReverseDiffusionPredictor).
        """
        return x, x


predictor_dict = dict(
    euler_maruyama=EulerMaruyamaPredictor,
    reverse_diffusion=ReverseDiffusionPredictor,
    none=NonePredictor,
)
