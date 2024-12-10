"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from
https://github.com/yang-song/score_sde_pytorch
and
https://github.com/sp-uhh/sgmse
"""

import abc
import warnings

import numpy as np
import torch


class SDE(abc.ABC):
    """
    Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

This module provides an abstract base class for Stochastic Differential
Equations (SDEs) and implementations for specific types of SDEs, such as 
Ornstein-Uhlenbeck Variance Exploding (OUVESDE) and Ornstein-Uhlenbeck 
Variance Preserving (OUVPSDE) SDEs. The SDE class includes methods for 
defining the dynamics of the SDE, marginal probabilities, and sampling 
from the prior distribution.

Taken and adapted from:
- https://github.com/yang-song/score_sde_pytorch
- https://github.com/sp-uhh/sgmse

Classes:
    SDE: Abstract class for SDEs.
    OUVESDE: Implementation of the Ornstein-Uhlenbeck Variance Exploding SDE.
    OUVPSDE: Implementation of the Ornstein-Uhlenbeck Variance Preserving SDE.

Usage:
    To create a specific SDE, instantiate one of the subclasses (e.g., 
    OUVESDE or OUVPSDE) and use its methods for sampling and computing 
    probabilities.

Attributes:
    N (int): Number of discretization time steps.
    
Args:
    N: Number of discretization time steps.

Examples:
    >>> ouvesde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
    >>> x = torch.randn(10, 3, 32, 32)  # Example input tensor
    >>> t = torch.tensor(0.5)  # Example time step
    >>> y = torch.randn(10, 3, 32, 32)  # Example steady-state mean
    >>> drift, diffusion = ouvesde.sde(x, t, y)
    >>> mean, std = ouvesde.marginal_prob(x, t, y)
    >>> sample = ouvesde.prior_sampling((10, 3, 32, 32), y)

Note:
    The "steady-state mean" `y` must be provided as an argument to the 
    methods which require it (e.g., `sde` or `marginal_prob`).

Todo:
    Implement the `prior_logp` method for OU SDE.
    """

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    This module provides abstract classes for Stochastic Differential Equations (SDEs),
    including Reverse SDEs and Variance Exploding/Preserving SDEs. It is adapted from
    the following repositories:
    - https://github.com/yang-song/score_sde_pytorch
    - https://github.com/sp-uhh/sgmse

    Attributes:
        N (int): Number of discretization time steps.

    Args:
        N (int): Number of discretization time steps.

    Returns:
        float: End time of the SDE.

    Yields:
        None

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.

    Examples:
        sde = MySDEClass(N=1000)
        end_time = sde.T
        drift, diffusion = sde.sde(x, t, *args)
        mean, std = sde.marginal_prob(x, t, *args)
        sample = sde.prior_sampling(shape, *args)
        log_density = sde.prior_logp(z)

    Note:
        The SDE class is abstract and must be subclassed to implement specific
        SDE behaviors.

    Todo:
        Implement prior_logp for specific SDE classes where applicable.
        """
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    This module contains abstract classes and implementations for Stochastic
    Differential Equations (SDEs), including reverse SDEs and variance
    exploding/preserving SDEs. These classes are designed for use with mini-batch
    inputs in machine learning applications.

    The code is adapted from:
    - https://github.com/yang-song/score_sde_pytorch
    - https://github.com/sp-uhh/sgmse

    Classes:
        SDE: Abstract base class for SDEs.
        OUVESDE: Implementation of an Ornstein-Uhlenbeck Variance Exploding SDE.
        OUVPSDE: Implementation of an Ornstein-Uhlenbeck Variance Preserving SDE.

    Notes:
        The "steady-state mean" `y` is not provided at construction but must be
        supplied as an argument to methods that require it (e.g., `sde` or
        `marginal_prob`).

    Examples:
        # Creating an instance of OUVESDE
        ouvesde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
        # Sampling from the prior distribution
        x_T = ouvesde.prior_sampling(shape=(10, 3, 32, 32), y=torch.zeros((10, 3, 32, 32)))

        # Creating an instance of OUVPSDE
        ouvpsde = OUVPSDE(beta_min=0.1, beta_max=0.5, stiffness=1, N=1000)
        # Obtaining marginal probabilities
        mean, std = ouvpsde.marginal_prob(x0=torch.zeros((10, 3, 32, 32)), t=0.5, y=torch.ones((10, 3, 32, 32)))
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """
        Parameters to determine the marginal distribution of

    the SDE, $p_t(x|args)$.

    Args:
        x: A tensor representing the initial state of the system.
        t: A float representing the time step at which to evaluate the 
           marginal distribution (from 0 to `self.T`).
        y: A tensor representing the steady-state mean that influences 
           the marginal distribution.

    Returns:
        A tuple containing:
            - mean: The expected value of the state at time `t`.
            - std: The standard deviation of the state at time `t`.

    Examples:
        >>> sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
        >>> mean, std = sde.marginal_prob(x0=torch.tensor([0.0]), t=0.5, y=1.0)
        >>> print(mean)
        >>> print(std)

    Note:
        The "steady-state mean" `y` must be provided as an argument to 
        this method.
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """
        Generate one sample from the prior distribution.

        This method samples from the prior distribution, denoted as 
        $p_T(x|args)$, with a specified output shape. It allows 
        for generating latent codes from the learned distribution at 
        the end time `T`.

        Args:
            shape (tuple): The desired shape of the generated sample.
            *args: Additional arguments specific to the SDE implementation.

        Returns:
            torch.Tensor: A sample drawn from the prior distribution with the 
            specified shape.

        Raises:
            UserWarning: If the provided shape does not match the shape of 
            the given `y`.

        Examples:
            >>> sde = OUVESDE()
            >>> sample = sde.prior_sampling((10, 3, 32, 32), y=torch.randn(10, 3, 32, 32))
            >>> print(sample.shape)
            torch.Size([10, 3, 32, 32])
        """
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """
        Compute log-density of the prior distribution.

    This function is useful for computing the log-likelihood via the 
    probability flow ODE. It should be implemented in subclasses of SDE 
    to provide the log-density of the prior distribution given a latent 
    code.

    Args:
        z: A tensor representing the latent code for which the log-density 
           is to be computed.

    Returns:
        log probability density: A tensor representing the log-density 
        of the prior distribution evaluated at the given latent code.

    Raises:
        NotImplementedError: If this method is not implemented in a 
        subclass.

    Examples:
        # Example usage in a subclass:
        class MySDE(SDE):
            def prior_logp(self, z):
                # Custom implementation for computing log-density
                return -0.5 * torch.sum(z ** 2)

        sde = MySDE(N=1000)
        latent_code = torch.randn(10, 3)  # Example latent code
        log_density = sde.prior_logp(latent_code)
        print(log_density)  # Output the log-density
        """
        pass

    def discretize(self, x, t, *args):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    This method is useful for reverse diffusion sampling and probability flow 
    sampling. It defaults to the Euler-Maruyama discretization method.

    Args:
        x (torch.Tensor): A tensor representing the state at time `t`.
        t (torch.Tensor): A torch float representing the time step (from 0 to 
                          `self.T`).
        *args: Additional arguments that may be required by the specific SDE 
                implementation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - f (torch.Tensor): The drift term scaled by the time step.
            - G (torch.Tensor): The diffusion term scaled by the square root 
              of the time step.

    Examples:
        >>> sde = MySDEClass(N=100)  # Replace with actual SDE class
        >>> x = torch.tensor([0.0])
        >>> t = torch.tensor(0.5)
        >>> f, G = sde.discretize(x, t)
        >>> print(f, G)

    Note:
        Ensure that the time `t` is within the range [0, self.T] for valid 
        results.
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    Taken and adapted from
    https://github.com/yang-song/score_sde_pytorch
    and
    https://github.com/sp-uhh/sgmse

    Attributes:
        N (int): Number of discretization time steps.

    Args:
        N (int): Number of discretization time steps.

    Returns:
        RSDE: An instance of the reverse-time SDE/ODE.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in
        subclasses.

    Examples:
        # Creating a reverse-time SDE with a score model
        reverse_sde = sde_instance.reverse(score_model, probability_flow=True)
        
        # Using the reverse SDE to generate samples
        samples = reverse_sde.prior_sampling(shape=(100, 3, 32, 32), y=some_tensor)

    Note:
        The reverse method constructs a reverse-time SDE/ODE that can be used
        for sampling and probabilistic inference.

    Todo:
        Implement specific methods for the derived classes to handle
        the unique behaviors of different SDEs.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = (
                    rsde_parts["total_drift"],
                    rsde_parts["diffusion"],
                )
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = (
                    -sde_diffusion[:, None, None, None] ** 2
                    * score
                    * (0.5 if self.probability_flow else 1.0)
                )
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )
                total_drift = sde_drift + score_drift
                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse

                diffusion sampler.
                """
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, *args) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

This module provides an abstract base class for Stochastic Differential
Equations (SDEs) and concrete implementations for specific SDE types,
including Ornstein-Uhlenbeck Variance Exploding SDE (OUVESDE) and
Ornstein-Uhlenbeck Variance Preserving SDE (OUVPSDE).

Taken and adapted from:
- https://github.com/yang-song/score_sde_pytorch
- https://github.com/sp-uhh/sgmse

Attributes:
    N: Number of discretization time steps.

Args:
    N (int): The number of discretization time steps for the SDE.

Returns:
    None

Examples:
    # Example usage of creating an instance of an SDE subclass
    ouvesde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
    drift, diffusion = ouvesde.sde(x, t, y)

Note:
    This module is intended for advanced users familiar with SDEs and
    their applications in probabilistic modeling and sampling.

Todo:
    - Implement `prior_logp` method for OU SDE.
        """
        pass


class OUVESDE(SDE):
    """
    Construct an Ornstein-Uhlenbeck Variance Exploding Stochastic Differential 
    Equation (SDE).

    This SDE is characterized by the following dynamics:

        dx = -theta * (y - x) dt + sigma(t) dw

    where:

        sigma(t) = sigma_min * (sigma_max/sigma_min)^t * 
        sqrt(2 * log(sigma_max/sigma_min))

    The "steady-state mean" `y` must be provided as an argument to the methods
    requiring it (e.g., `sde` or `marginal_prob`).

    Attributes:
        theta (float): Stiffness parameter.
        sigma_min (float): Minimum value for sigma.
        sigma_max (float): Maximum value for sigma.
        N (int): Number of discretization steps.
        logsig (float): Logarithm of the ratio between sigma_max and sigma_min.

    Args:
        theta (float): Stiffness parameter.
        sigma_min (float): Smallest sigma.
        sigma_max (float): Largest sigma.
        N (int): Number of discretization steps.

    Examples:
        >>> ouvesde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
        >>> x = torch.tensor([0.0])
        >>> y = torch.tensor([1.0])
        >>> t = torch.tensor(0.5)
        >>> drift, diffusion = ouvesde.sde(x, t, y)
        >>> mean, std = ouvesde.marginal_prob(x, t, y)

    Raises:
        NotImplementedError: If `prior_logp` is called as it is not implemented.
    """
    def __init__(
        self, theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000, **ignored_kwargs
    ):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction,
        but must rather be given as an argument to the methods
        which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from
https://github.com/yang-song/score_sde_pytorch
and
https://github.com/sp-uhh/sgmse
        """
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    Taken and adapted from
    https://github.com/yang-song/score_sde_pytorch
    and
    https://github.com/sp-uhh/sgmse

    This module provides abstract classes and implementations for stochastic 
    differential equations (SDEs) that are used in various probabilistic models. 
    The classes are designed to handle different types of SDEs, including 
    Ornstein-Uhlenbeck processes with variance exploding and variance preserving 
    properties.

    Classes:
        SDE: Abstract base class for stochastic differential equations.
        OUVESDE: Implementation of the Ornstein-Uhlenbeck Variance Exploding SDE.
        OUVPSDE: Implementation of the Ornstein-Uhlenbeck Variance Preserving SDE.

    Usage:
        These classes can be used as base classes for specific SDE implementations 
        where the drift and diffusion functions can be defined. The user can 
        instantiate these classes and call their methods to simulate SDE behavior 
        or to compute marginal probabilities.
        """
        return 1

    def sde(self, x, t, y):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    This module contains abstract classes for Stochastic Differential Equations (SDEs),
    including Reverse SDE and Variance Exploding/Preserving SDEs. It has been adapted
    from the following repositories:
    - https://github.com/yang-song/score_sde_pytorch
    - https://github.com/sp-uhh/sgmse

    Classes:
        SDE: Abstract class for SDEs, designed for mini-batch processing.
        OUVESDE: Implements an Ornstein-Uhlenbeck Variance Exploding SDE.
        OUVPSDE: Implements an Ornstein-Uhlenbeck Variance Preserving SDE.

    Usage:
        You can create an instance of OUVESDE or OUVPSDE and call their methods
        to perform operations like sampling, computing marginal probabilities, etc.

    Example:
        # Create an instance of OUVESDE
        sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)

        # Sample from the prior distribution
        sample = sde.prior_sampling(shape=(10, 3), y=torch.tensor([[0.0, 0.0, 0.0]]))

        # Compute marginal probability
        mean, std = sde.marginal_prob(x0=torch.tensor([[0.0, 0.0, 0.0]]), t=0.5, y=torch.tensor([[1.0, 1.0, 1.0]]))

    Note:
        The "steady-state mean" `y` is not provided at construction but must be given
        as an argument to methods that require it (e.g., `sde` or `marginal_prob`).
        """
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end
        # affect the perturbation kernel standard deviation. this can be understood
        # from solving the integral of [exp(2s) * g(s)^2] from s=0 to t with
        # g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the
        # integral solution unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations,
        # after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        """
        Compute the marginal probability distribution of the SDE.

        This method calculates the mean and standard deviation of the 
        marginal distribution at a given time `t` for the state variable 
        `x`, conditioned on the steady-state mean `y`. The marginal 
        distribution is defined by the Ornstein-Uhlenbeck process 
        parameters.

        Args:
            x0: Initial state variable (tensor).
            t: Time at which to evaluate the marginal distribution (float).
            y: Steady-state mean (tensor).

        Returns:
            A tuple containing:
                - mean (tensor): The mean of the marginal distribution.
                - std (tensor): The standard deviation of the marginal 
                  distribution.

        Examples:
            >>> sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5)
            >>> mean, std = sde.marginal_prob(x0=torch.tensor([0.0]), 
            ...                                 t=0.5, 
            ...                                 y=torch.tensor([1.0]))
            >>> print(mean, std)

        Note:
            The mean and standard deviation depend on the parameters 
            `theta`, `sigma_min`, `sigma_max`, and the input `y`.
        """
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        """
        Generate one sample from the prior distribution, $p_T(x|args)$.

    This method generates a sample from the prior distribution defined by
    the Ornstein-Uhlenbeck process. It adds Gaussian noise to the input
    `y`, scaled by the standard deviation computed at time `T`.

    Args:
        shape: Desired shape of the output sample. If it does not match
            the shape of `y`, a warning is issued and the shape of `y`
            is used instead.
        y: The steady-state mean around which the sample is generated.
            This should be a tensor of shape compatible with the output.

    Returns:
        A tensor of shape `shape` containing a sample from the prior
        distribution.

    Raises:
        Warning: If the target shape does not match the shape of `y`.

    Examples:
        >>> ouvesde = OUVESDE()
        >>> y = torch.zeros((10, 3, 32, 32))
        >>> sample = ouvesde.prior_sampling((10, 3, 32, 32), y)
        >>> print(sample.shape)
        torch.Size([10, 3, 32, 32])
        """
        if shape != y.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}!"
                "Ignoring target shape."
            )
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        """
        Compute log-density of the prior distribution.

        This method is useful for computing the log-likelihood via 
        probability flow ODE.

        Args:
            z: Latent code for which the log-density is to be computed.

        Returns:
            log probability density corresponding to the input latent code.

        Raises:
            NotImplementedError: This method is not yet implemented for the 
            Ornstein-Uhlenbeck Variance Exploding SDE.

        Examples:
            >>> sde = OUVESDE()
            >>> z = torch.tensor([0.5, 0.2])
            >>> log_density = sde.prior_logp(z)
            NotImplementedError: prior_logp for OU SDE not yet implemented!
        """
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


class OUVPSDE(SDE):
    """
    OUVPSDE class.

    !!! SGMSE authors observed instabilities around t=0.2. !!!

    Construct an Ornstein-Uhlenbeck Variance Preserving SDE:

    dx = -1/2 * beta(t) * stiffness * (y-x) dt + sqrt(beta(t)) * dw

    with

    beta(t) = beta_min + t(beta_max - beta_min)

    Note that the "steady-state mean" `y` is not provided at construction,
    but must rather be given as an argument to the methods which
    require it (e.g., `sde` or `marginal_prob`).

    Attributes:
        beta_min (float): Smallest value for beta.
        beta_max (float): Largest value for beta.
        stiffness (float): Stiffness factor of the drift, default is 1.
        N (int): Number of discretization steps.

    Args:
        beta_min: Smallest beta value.
        beta_max: Largest beta value.
        stiffness: Stiffness factor of the drift. 1 by default.
        N: Number of discretization steps.

    Returns:
        None

    Examples:
        >>> ou_vpsde = OUVPSDE(beta_min=0.1, beta_max=0.5, stiffness=1, N=1000)
        >>> x0 = torch.randn(10, 3, 32, 32)  # Example input
        >>> y = torch.randn(10, 3, 32, 32)  # Steady-state mean
        >>> t = torch.tensor(0.5)  # Time step
        >>> mean, std = ou_vpsde.marginal_prob(x0, t, y)
        
    Raises:
        NotImplementedError: If prior_logp is called as it is not implemented.
    """
    def __init__(self, beta_min, beta_max, stiffness=1, N=1000, **ignored_kwargs):
        """OUVPSDE class.

        !!! SGMSE authors observed instabilities around t=0.2. !!!

        Construct an Ornstein-Uhlenbeck Variance Preserving SDE:

        dx = -1/2 * beta(t) * stiffness * (y-x) dt + sqrt(beta(t)) * dw

        with

        beta(t) = beta_min + t(beta_max - beta_min)

        Note that the "steady-state mean" `y` is not provided at construction,
        but must rather be given as an argument to the methods which
        require it (e.g., `sde` or `marginal_prob`).

        Args:
            beta_min: smallest sigma.
            beta_max: largest sigma.
            stiffness: stiffness factor of the drift. 1 by default.
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.stiffness = stiffness
        self.N = N

    def copy(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    Taken and adapted from
    https://github.com/yang-song/score_sde_pytorch
    and
    https://github.com/sp-uhh/sgmse

    This module defines the abstract base class `SDE` for stochastic differential 
    equations (SDEs) and its derived classes for specific types of SDEs. The main 
    purpose of these classes is to provide a framework for implementing and 
    working with SDEs, including the ability to discretize them, generate 
    samples from their distributions, and compute marginal probabilities.

    The `SDE` class is an abstract class that defines the necessary methods and 
    properties that must be implemented by any concrete SDE subclass. 

    The `OUVESDE` and `OUVPSDE` classes are concrete implementations of 
    the Ornstein-Uhlenbeck SDEs with variance exploding and variance preserving 
    characteristics, respectively.

    Attributes:
        N: number of discretization time steps.

    Args:
        N: int, number of discretization time steps for the SDE.

    Returns:
        A class that implements the specified SDE functionality.

    Yields:
        None

    Raises:
        NotImplementedError: If any of the abstract methods are not implemented
        by a subclass.

    Examples:
        # Example usage of the OUVPSDE class
        sde = OUVPSDE(beta_min=0.1, beta_max=1.0, stiffness=1.0, N=1000)
        x = torch.tensor([[0.0]])
        y = torch.tensor([[1.0]])
        t = torch.tensor([0.5])
        drift, diffusion = sde.sde(x, t, y)
        print(drift, diffusion)
        
        # Example usage of the OUVESDE class
        ouve_sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=1000)
        x0 = torch.tensor([[0.0]])
        y = torch.tensor([[1.0]])
        t = torch.tensor([0.5])
        mean, std = ouve_sde.marginal_prob(x0, t, y)
        print(mean, std)
        """
        return OUVPSDE(self.beta_min, self.beta_max, self.stiffness, N=self.N)

    @property
    def T(self):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    This module defines abstract classes and methods for Stochastic Differential
    Equations (SDEs), including the Reverse SDE and Variance Exploding/Preserving
    SDEs. It is adapted from the works available at:
    https://github.com/yang-song/score_sde_pytorch and
    https://github.com/sp-uhh/sgmse.

    Classes:
        SDE: Abstract class for Stochastic Differential Equations.
        OUVESDE: Ornstein-Uhlenbeck Variance Exploding SDE class.
        OUVPSDE: Ornstein-Uhlenbeck Variance Preserving SDE class.

    Functions:
        batch_broadcast: Broadcasts a tensor over all dimensions of another tensor,
        except the batch dimension.

    Attributes:
        T: End time of the SDE.
        
    Args:
        N: Number of discretization time steps.

    Raises:
        NotImplementedError: Raised if a method that is not implemented is called.

    Examples:
        # Creating an instance of OUVPSDE
        ouvpsde = OUVPSDE(beta_min=0.1, beta_max=1.0, stiffness=1.0, N=1000)
        
        # Using the instance to sample from the prior distribution
        sample_shape = (10, 3, 64, 64)  # Example shape
        y = torch.zeros(sample_shape)  # Example steady-state mean
        sample = ouvpsde.prior_sampling(shape=sample_shape, y=y)
        
        # Getting marginal probabilities
        mean, std = ouvpsde.marginal_prob(x0=torch.zeros(sample_shape), t=0.5, y=y)
        """
        return 1

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t, y):
        """
        Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

    Taken and adapted from:
    https://github.com/yang-song/score_sde_pytorch
    and
    https://github.com/sp-uhh/sgmse

    This module defines the abstract base class for Stochastic Differential Equations
    (SDEs) and implements specific SDE types, including the Ornstein-Uhlenbeck 
    Variance Exploding SDE (OUVESDE) and Ornstein-Uhlenbeck Variance Preserving SDE 
    (OUVPSDE). These classes facilitate the modeling and simulation of SDEs, 
    particularly for applications in generative modeling and diffusion processes.

    The SDE class serves as an abstract base for specific implementations, defining 
    the core methods that must be implemented by derived classes, such as 
    `marginal_prob`, `prior_sampling`, and `prior_logp`. It also provides methods 
    for discretizing the SDE and creating reverse-time SDEs.

    Attributes:
        N (int): Number of discretization time steps.

    Args:
        N (int): Number of discretization time steps.

    Examples:
        # Example of using the OUVPSDE class
        sde = OUVPSDE(beta_min=0.1, beta_max=1.0, stiffness=1, N=1000)
        x0 = torch.randn(10, 3, 32, 32)  # Example input tensor
        y = torch.randn(10, 3, 32, 32)   # Example steady-state mean
        t = torch.tensor(0.5)             # Example time
        mean, std = sde.marginal_prob(x0, t, y)

        # Example of reverse-time SDE creation
        score_model = lambda x, t, *args: -x  # Dummy score model
        reverse_sde = sde.reverse(score_model)

    Note:
        Ensure to implement the abstract methods when extending the SDE class.

    Todo:
        - Implement prior_logp for OU SDEs in derived classes.
        """
        drift = 0.5 * self.stiffness * batch_broadcast(self._beta(t), y) * (y - x)
        diffusion = torch.sqrt(self._beta(t))
        return drift, diffusion

    def _mean(self, x0, t, y):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        x0y_fac = torch.exp(-0.25 * s * t * (t * (b1 - b0) + 2 * b0))[
            :, None, None, None
        ]
        return y + x0y_fac * (x0 - y)

    def _std(self, t):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        return (1 - torch.exp(-0.5 * s * t * (t * (b1 - b0) + 2 * b0))) / s

    def marginal_prob(self, x0, t, y):
        """
        Calculate the marginal distribution of the SDE.

        This method computes the parameters that define the marginal 
        distribution of the stochastic differential equation (SDE), 
        denoted as \( p_t(x|args) \). The specific implementation 
        should return the mean and standard deviation of the 
        marginal distribution.

        Args:
            x: A tensor representing the current state.
            t: A float representing the time step.
            *args: Additional arguments required for the calculation, 
                such as the steady-state mean `y`.

        Returns:
            A tuple containing:
                - mean: The mean of the marginal distribution.
                - std: The standard deviation of the marginal distribution.

        Examples:
            >>> sde = OUVPSDE(beta_min=0.1, beta_max=0.5)
            >>> mean, std = sde.marginal_prob(x=torch.tensor([0.0]), t=0.5, y=1.0)
            >>> print(mean, std)

        Note:
            The steady-state mean `y` must be provided as an argument 
            to this method.
        """
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        """
        Generate one sample from the prior distribution,

    $p_T(x|args)$ with shape `shape`.

    The method generates a sample at the end time `T` of the SDE, which can be
    influenced by the steady-state mean `y`. If the provided shape does not
    match the shape of `y`, a warning will be issued, and the target shape will
    be ignored.

    Args:
        shape: The desired shape of the generated sample.
        y: The steady-state mean to influence the sample generation.

    Returns:
        A tensor representing a sample drawn from the prior distribution.

    Raises:
        UserWarning: If the provided shape does not match the shape of `y`.

    Examples:
        >>> sde = OUVPSDE(beta_min=0.1, beta_max=0.5)
        >>> y = torch.randn(10, 3, 32, 32)  # Example steady-state mean
        >>> sample = sde.prior_sampling((10, 3, 32, 32), y)
        >>> print(sample.shape)  # Output: torch.Size([10, 3, 32, 32])
        """
        if shape != y.shape:
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}!"
                "Ignoring target shape."
            )
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        """
        Compute log-density of the prior distribution.

        This method is useful for computing the log-likelihood via 
        probability flow ODE. It should be implemented in subclasses 
        to provide the actual log-density computation for the prior 
        distribution.

        Args:
            z: A tensor representing the latent code for which the log-density 
               is to be computed.

        Returns:
            log probability density: A scalar tensor representing the log-density 
            of the prior distribution evaluated at the given latent code.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            # Assuming `model` is an instance of a subclass that implements 
            # `prior_logp` and `latent_code` is a tensor representing the 
            # latent variable.
            log_density = model.prior_logp(latent_code)
        """
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


def batch_broadcast(a, x):
    """
    Broadcasts a over all dimensions of x, except the batch dimension.

    This function ensures that the tensor `a` is broadcasted across all dimensions
    of the tensor `x`, while preserving the batch dimension. The batch dimension 
    must match between `a` and `x`, or `a` must be a scalar.

    Args:
        a (torch.Tensor): The tensor to be broadcasted.
        x (torch.Tensor): The tensor over which `a` will be broadcasted.

    Returns:
        torch.Tensor: A tensor of the same shape as `x` with `a` broadcasted 
        across its dimensions.

    Raises:
        ValueError: If `a` has more than one effective dimension after squeezing,
        or if the batch dimensions of `a` and `x` do not match.

    Examples:
        >>> import torch
        >>> a = torch.tensor([1.0, 2.0, 3.0])  # shape: (3,)
        >>> x = torch.zeros((5, 4, 2))  # shape: (5, 4, 2)
        >>> result = batch_broadcast(a, x)
        >>> result.shape
        torch.Size([5, 4, 3])

        >>> a = torch.tensor([1.0])  # shape: (1,)
        >>> x = torch.zeros((5, 4, 2))  # shape: (5, 4, 2)
        >>> result = batch_broadcast(a, x)
        >>> result.shape
        torch.Size([5, 4, 1])

        >>> a = torch.tensor([1.0, 2.0])  # shape: (2,)
        >>> x = torch.zeros((5, 4, 2))  # shape: (5, 4, 2)
        >>> result = batch_broadcast(a, x)
        >>> result.shape
        torch.Size([5, 4, 2])
    """

    if len(a.shape) != 1:
        a = a.squeeze()
        if len(a.shape) != 1:
            raise ValueError(
                f"Don't know how to batch-broadcast tensor `a` "
                f"with more than one effective dimension (shape {a.shape})"
            )

    if a.shape[0] != x.shape[0] and a.shape[0] != 1:
        raise ValueError(
            f"Don't know how to batch-broadcast shape {a.shape} over {x.shape} "
            "as the batch dimension is not matching"
        )

    out = a.view((x.shape[0], *(1 for _ in range(len(x.shape) - 1))))
    return out
