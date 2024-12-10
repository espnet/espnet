# Adapted from https://github.com/yang-song/score_sde_pytorch/
# and https://github.com/sp-uhh/sgmse
import abc

import torch

import espnet2.enh.diffusion.sdes as sdes


class Corrector(abc.ABC):
    """
    The abstract class for a corrector algorithm.

    This class serves as a base for implementing various corrector algorithms
    used in diffusion models. It requires subclasses to implement the
    `update_fn` method, which performs a single update step of the
    corrector algorithm.

    Attributes:
        rsde: The reverse SDE obtained from the provided score function.
        score_fn: A function that estimates the score (gradient of log
            probability) of the data.
        snr: Signal-to-noise ratio used in the update step.
        n_steps: Number of steps to perform in the update process.

    Args:
        sde: The stochastic differential equation to be used.
        score_fn: A callable function to estimate the score.
        snr: A float representing the signal-to-noise ratio.
        n_steps: An integer indicating the number of steps for the update.

    Raises:
        NotImplementedError: If the subclass does not implement the
            `update_fn` method.

    Examples:
        # Example subclass implementation
        class MyCorrector(Corrector):
            def update_fn(self, x, t, *args):
                # Custom update logic here
                pass
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """
        One update of the corrector.

        This method performs a single update step of the corrector algorithm,
        adjusting the current state based on the provided inputs and the
        specific corrector implementation.

        Args:
            x (torch.Tensor): A PyTorch tensor representing the current state.
            t (torch.Tensor): A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): A PyTorch tensor of the next state.
                - x_mean (torch.Tensor): A PyTorch tensor of the next state without
                  random noise. Useful for denoising.

        Examples:
            # Example usage of update_fn
            current_state = torch.randn(1, 3, 32, 32)  # Example current state
            current_time = torch.tensor(0.5)  # Example current time step
            next_state, denoised_state = corrector_instance.update_fn(current_state, current_time)

        Note:
            The specific behavior of the update step depends on the concrete
            implementation of the Corrector subclass.
        """
        pass


class LangevinCorrector(Corrector):
    """
    Langevin Corrector for score-based diffusion models.

    This class implements the Langevin Corrector algorithm, which is used in
    score-based generative models to iteratively refine samples by using
    Langevin dynamics. It updates the state based on the gradients from the
    score function and adds noise to facilitate exploration of the sample
    space.

    Attributes:
        score_fn (callable): A function that computes the score at a given
            state and time.
        snr (float): The signal-to-noise ratio used to scale the updates.
        n_steps (int): The number of update steps to perform in each call.

    Args:
        sde: The stochastic differential equation object.
        score_fn: The score function used to compute gradients.
        snr: The signal-to-noise ratio.
        n_steps: The number of steps for the Langevin dynamics update.

    Returns:
        tuple: A tuple containing:
            - x (torch.Tensor): The updated state after applying the Langevin
              dynamics.
            - x_mean (torch.Tensor): The mean state without random noise,
              useful for denoising.

    Examples:
        >>> import torch
        >>> sde = ...  # Define your SDE here
        >>> score_fn = ...  # Define your score function here
        >>> corrector = LangevinCorrector(sde, score_fn, snr=1.0, n_steps=10)
        >>> x_init = torch.randn(1, 3, 64, 64)  # Example initial state
        >>> t = torch.tensor(0.5)  # Example time step
        >>> x_updated, x_mean = corrector.update_fn(x_init, t)

    Note:
        This corrector assumes that the score function is well-defined and
        can handle the inputs provided.

    Raises:
        NotImplementedError: If the score function or SDE is not compatible
        with the Langevin dynamics.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.snr = snr

    def update_fn(self, x, t, *args):
        """
            One update of the corrector.

        This method performs a single update step of the corrector algorithm,
        modifying the current state `x` based on the specified time step `t`.
        The update involves computing the gradient of the score function,
        generating noise, and calculating the new state.

        Args:
            x (torch.Tensor): A PyTorch tensor representing the current state.
            t (torch.Tensor): A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): A PyTorch tensor of the next state.
                - x_mean (torch.Tensor): A PyTorch tensor representing the next state
                  without random noise. Useful for denoising.

        Examples:
            >>> corrector = LangevinCorrector(sde, score_fn, snr, n_steps)
            >>> next_state, denoised_state = corrector.update_fn(current_state, time_step)

        Note:
            This method is expected to be overridden in subclasses to provide
            specific update logic based on the corrector algorithm being implemented.
        """
        target_snr = self.snr
        for _ in range(self.n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = ((target_snr * noise_norm / grad_norm) ** 2 * 2).unsqueeze(0)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """
    The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    This class implements the Annealed Langevin Dynamics algorithm as a
    corrector for the diffusion sampling process. It is specifically designed
    for use with Ornstein-Uhlenbeck (OU) processes, leveraging the score
    function to iteratively refine the state estimate.

    Attributes:
        sde: The stochastic differential equation (SDE) used for the process.
        score_fn: The function that estimates the score of the data.
        snr: The signal-to-noise ratio used in the dynamics.
        n_steps: The number of steps to take in the Langevin update.

    Args:
        sde: An instance of a stochastic differential equation (SDE) class,
             specifically expected to be an OU process.
        score_fn: A callable that computes the score (gradient of the log
                  probability) of the current state.
        snr: A float representing the desired signal-to-noise ratio.
        n_steps: An integer representing the number of Langevin steps to
                 perform in the update.

    Raises:
        NotImplementedError: If the provided SDE is not an instance of
                             `sdes.OUVESDE`.

    Returns:
        A tuple of two PyTorch tensors:
            - x: The updated state tensor after applying the Langevin dynamics.
            - x_mean: The denoised state tensor (mean state) without
                       random noise, useful for further processing.

    Examples:
        # Assuming `sde`, `score_fn`, `initial_state`, and `time_step` are defined
        ald_corrector = AnnealedLangevinDynamics(sde, score_fn, snr=1.0, n_steps=10)
        updated_state, denoised_state = ald_corrector.update_fn(initial_state, time_step)

    Note:
        The algorithm relies on the SDE's ability to compute marginal probabilities
        and assumes that the score function is properly defined for the given
        state and time step.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, (sdes.OUVESDE,)):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, t, *args):
        """
        One update of the corrector.

        This method performs a single update step of the corrector algorithm,
        adjusting the current state `x` based on the score function and the
        provided time step `t`. The update is performed over a specified number
        of steps (`n_steps`), incorporating noise and the signal-to-noise ratio
        (SNR) to achieve the desired state.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU
                processes.

        Returns:
            A tuple containing:
                - x: A PyTorch tensor of the next state after the update.
                - x_mean: A PyTorch tensor representing the next state
                  without random noise, useful for denoising.

        Examples:
            # Example usage:
            # Assuming `score_fn` is defined and returns a tensor of gradients
            # for the current state `x` at time `t`, and `sde` is an instance
            # of the SDE class.
            x_next, x_mean_next = self.update_fn(x_current, t_current)

        Note:
            The update step is influenced by the `score_fn`, which is expected
            to compute the gradient of the log probability of the data. The
            noise added during the update is sampled from a standard normal
            distribution.

        Raises:
            ValueError: If the input tensors `x` and `t` are not of the
                expected shape or type.
        """
        n_steps = self.n_steps
        target_snr = self.snr
        std = self.sde.marginal_prob(x, t, *args)[1]

        for _ in range(n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """
        NoneCorrector is an implementation of the Corrector class that performs no
    operations during the update phase. It is essentially a placeholder corrector
    that can be used in scenarios where no correction is needed.

    Attributes:
        snr (float): The signal-to-noise ratio, set to 0.
        n_steps (int): The number of steps, set to 0.

    Args:
        *args: Additional positional arguments (not used).
        **kwargs: Additional keyword arguments (not used).

    Returns:
        x (torch.Tensor): The input tensor unchanged.
        x (torch.Tensor): The input tensor unchanged.

    Examples:
        >>> corrector = NoneCorrector()
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t = torch.tensor([0.5])
        >>> next_state, next_mean = corrector.update_fn(x, t)
        >>> print(next_state)
        tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> print(next_mean)
        tensor([[1.0, 2.0], [3.0, 4.0]])

    Note:
        This corrector is useful for situations where a corrector interface is
        required but no actual correction is needed.

    Todo:
        Implement a logging mechanism to track when this corrector is used.
    """

    def __init__(self, *args, **kwargs):
        self.snr = 0
        self.n_steps = 0
        pass

    def update_fn(self, x, t, *args):
        """
            An empty corrector that does nothing.

        This corrector is used when no correction is needed. It simply returns the
        input state as the output state without any modifications.

        Attributes:
            snr (float): Signal-to-noise ratio, initialized to 0.
            n_steps (int): Number of steps for the correction process, initialized to 0.

        Args:
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): The input state tensor, unchanged.
                - x (torch.Tensor): The input state tensor, unchanged.

        Examples:
            >>> corrector = NoneCorrector()
            >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> t = torch.tensor([0.5])
            >>> updated_x, x_mean = corrector.update_fn(x, t)
            >>> print(updated_x)
            tensor([[1., 2.],
                    [3., 4.]])
            >>> print(x_mean)
            tensor([[1., 2.],
                    [3., 4.]])
        """
        return x, x


corrector_dict = dict(
    langevin=LangevinCorrector, ald=AnnealedLangevinDynamics, none=NoneCorrector
)
