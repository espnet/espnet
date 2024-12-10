# The implementation is based on:
# https://github.com/sp-uhh/sgmse
# Licensed under MIT


import math

import torch

import espnet2.enh.diffusion.sampling as sampling
from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion
from espnet2.enh.diffusion.sdes import OUVESDE, OUVPSDE, SDE
from espnet2.enh.layers.dcunet import DCUNet
from espnet2.enh.layers.ncsnpp import NCSNpp
from espnet2.train.class_choices import ClassChoices

score_choices = ClassChoices(
    name="score_model",
    classes=dict(dcunet=DCUNet, ncsnpp=NCSNpp),
    type_check=torch.nn.Module,
    default=None,
)

sde_choices = ClassChoices(
    name="sde",
    classes=dict(
        ouve=OUVESDE,
        ouvp=OUVPSDE,
    ),
    type_check=SDE,
    default="ouve",
)


class ScoreModel(AbsDiffusion):
    """
    Score-based model for enhancing audio features using diffusion processes.

    This class implements a score-based model for audio enhancement based on
    diffusion processes. It leverages neural networks to estimate the score
    function and perform sampling using either Predictor-Corrector (PC) or
    Ordinary Differential Equation (ODE) methods.

    Attributes:
        dnn (torch.nn.Module): Deep neural network used for score estimation.
        sde (SDE): Stochastic differential equation used for diffusion.
        loss_type (str): Type of loss function to use ('mse' or 'mae').
        t_eps (float): Small constant to prevent division by zero.

    Args:
        **kwargs: Keyword arguments to configure the score model and SDE.

    Raises:
        ValueError: If an invalid sampler type is specified in the enhance method.

    Examples:
        # Creating a score model
        score_model = ScoreModel(
            score_model='dcunet',
            score_model_conf={'param1': value1},
            sde='ouve',
            sde_conf={'param2': value2},
            loss_type='mse',
            t_eps=1e-2
        )

        # Enhancing a noisy spectrum
        enhanced_spectrum = score_model.enhance(
            noisy_specturm=torch.randn(4, 128, 256),
            sampler_type='pc',
            predictor='reverse_diffusion',
            corrector='ald',
            N=30,
            corrector_steps=1,
            snr=0.5
        )
    """

    def __init__(self, **kwargs):
        super().__init__()

        score_model = kwargs["score_model"]  # noqa
        score_model_class = score_choices.get_class(kwargs["score_model"])
        self.dnn = score_model_class(**kwargs["score_model_conf"])
        self.sde = sde_choices.get_class(kwargs["sde"])(**kwargs["sde_conf"])
        self.loss_type = getattr(kwargs, "loss_type", "mse")
        self.t_eps = getattr(kwargs, "t_eps", 3e-2)

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position
        # and mean over batch dim presumably only important for absolute
        # loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs
    ):
        """
            Retrieve a Predictor-Corrector sampler for the score model.

        This method creates a sampler that uses a predictor-corrector
        approach for sampling from the score model's diffusion process.
        It allows for optional minibatching to handle large input data
        more efficiently.

        Args:
            predictor_name (str): The name of the predictor to use.
            corrector_name (str): The name of the corrector to use.
            y (torch.Tensor): The input data tensor of shape [Batch, Channels, Time].
            N (int, optional): The number of sampling steps. If None, the default
                               from the SDE will be used.
            minibatch (int, optional): The size of the minibatches for sampling.
                                        If None, the full batch will be processed
                                        at once.
            **kwargs: Additional keyword arguments to be passed to the sampler.

        Returns:
            Callable: A sampling function that returns samples and the number
                      of function evaluations when called.

        Examples:
            >>> sampler = model.get_pc_sampler('reverse_diffusion', 'ald', y)
            >>> samples, nfe = sampler()

        Note:
            If minibatch is specified, the sampling will be performed in smaller
            chunks, which may help to manage memory usage for large input tensors.

        Raises:
            ValueError: If the provided predictor_name or corrector_name is invalid.
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self.score_fn,
                y=y,
                **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self.score_fn,
                        y=y_mini,
                        **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        """
        Retrieve an ODE sampler for generating samples based on the score model.

        This method provides an interface to obtain a sampler that uses
        Ordinary Differential Equations (ODE) to sample from the score
        model. It supports both batch and non-batch sampling.

        Args:
            y (torch.Tensor): Input tensor of shape [Batch, T, F] used as a
                starting point for the sampling process.
            N (int, optional): The number of sampling steps to perform. If
                None, the default value from the SDE instance is used.
            minibatch (int, optional): The size of the minibatch for
                batched sampling. If None, the entire input tensor is
                processed in one go.
            **kwargs: Additional keyword arguments passed to the sampler.

        Returns:
            Union[Tuple[torch.Tensor, List[int]], Callable]:
                If minibatch is None, returns a tuple containing the
                generated samples and the number of function evaluations
                (nfe). If minibatch is specified, returns a function
                that generates samples in batches.

        Examples:
            >>> model = ScoreModel(score_model='dcunet', sde='ouve')
            >>> y = torch.randn(10, 100, 64)  # Example input
            >>> sampler = model.get_ode_sampler(y)
            >>> samples, nfe = sampler()  # Generate samples

            >>> sampler_batched = model.get_ode_sampler(y, minibatch=5)
            >>> samples, nfe = sampler_batched()  # Generate samples in batches

        Note:
            The method uses the score function defined in the model to
            generate samples based on the input tensor y.
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(
                sde, self.score_fn, y=y, device=y.device, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self.score_fn, y=y_mini, device=y_mini.device, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def score_fn(self, x, t, y):
        """
            Compute the score function based on the input tensor.

        This function concatenates the input tensor `y` as an additional channel
        to the input tensor `x` and computes the score using the model's deep
        neural network (DNN). The score is the negative output of the DNN, which
        represents the gradient of the log probability density function.

        Args:
            x (torch.Tensor): The input tensor of shape [B, C, T] where B is the
                              batch size, C is the number of channels, and T is
                              the time dimension.
            t (torch.Tensor): The time tensor of shape [B] representing the time
                              steps at which the score is computed.
            y (torch.Tensor): The auxiliary tensor that is concatenated with `x`,
                              having the same number of channels.

        Returns:
            torch.Tensor: The computed score tensor of shape [B, C, T].

        Examples:
            >>> x = torch.randn(4, 1, 100)  # Batch of 4, 1 channel, 100 time steps
            >>> t = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Example time steps
            >>> y = torch.randn(4, 1, 100)  # Auxiliary tensor
            >>> score = score_fn(x, t, y)
            >>> print(score.shape)  # Should print: torch.Size([4, 1, 100])
        """
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def forward(
        self,
        feature_ref,
        feature_mix,
    ):
        """
            Score Model for Diffusion-based Signal Enhancement.

        This class implements a score model for enhancing signals using
        diffusion processes. It utilizes deep learning architectures to
        estimate the score function, which is essential for various
        sampling methods.

        Attributes:
            dnn (torch.nn.Module): Deep neural network used for score estimation.
            sde (SDE): Stochastic differential equation for the diffusion process.
            loss_type (str): Type of loss function used ('mse' or 'mae').
            t_eps (float): Small epsilon value for time steps.

        Args:
            **kwargs: Keyword arguments for model configuration, including:
                score_model (str): Name of the score model to use ('dcunet' or 'ncsnpp').
                score_model_conf (dict): Configuration for the score model.
                sde (str): Name of the SDE to use ('ouve' or 'ouvp').
                sde_conf (dict): Configuration for the SDE.
                loss_type (str, optional): Loss function type ('mse' or 'mae'). Default is 'mse'.
                t_eps (float, optional): Epsilon value for time steps. Default is 3e-2.

        Returns:
            None

        Raises:
            AssertionError: If the output shape of the score function does not match
            the expected shape.

        Examples:
            >>> model = ScoreModel(score_model='dcunet', score_model_conf={},
            ...                     sde='ouve', sde_conf={})
            >>> feature_ref = torch.randn(2, 10, 5)  # Example reference features
            >>> feature_mix = torch.randn(2, 10, 5)  # Example mixed features
            >>> loss = model.forward(feature_ref, feature_mix)
            >>> print(loss)

        Note:
            This implementation is based on the research from the repository:
            https://github.com/sp-uhh/sgmse and is licensed under the MIT license.

        Todo:
            - Implement additional loss functions.
            - Add more sampling methods for flexibility.
        """
        # feature_ref: B, T, F
        # feature_mix: B, T, F
        x = feature_ref.permute(0, 2, 1).unsqueeze(1)
        y = feature_mix.permute(0, 2, 1).unsqueeze(1)

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self.score_fn(perturbed_data, t, y)
        assert score.shape == x.shape, "Check the output shape of the score_fn."
        err = score * sigmas + z
        loss = self._loss(err)

        return loss

    def enhance(
        self,
        noisy_specturm,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        **kwargs
    ):
        """
            Enhance function.

        This method enhances a noisy spectrum by applying a specified sampling
        technique (Predictor-Corrector or ODE). It utilizes a trained score model
        to improve the quality of the input noisy spectrum.

        Args:
            noisy_specturm (torch.Tensor): Noisy feature in shape [Batch, T, F].
            sampler_type (str): Sampler type, either 'pc' for Predictor-Corrector
                                or 'ode' for ODE sampler.
            predictor (str): Name of the Predictor. Options include
                             'reverse_diffusion', 'euler_maruyama', or 'none'.
            corrector (str): Name of the Corrector. Options include
                             'langevin', 'ald', or 'none'.
            N (int): The number of reverse sampling steps.
            corrector_steps (int): Number of steps in the Corrector.
            snr (float): The Signal-to-Noise Ratio (SNR) to use for the corrector.
            **kwargs: Additional keyword arguments for the sampler.

        Returns:
            torch.Tensor: Enhanced feature in shape [Batch, T, F].

        Raises:
            ValueError: If an invalid sampler type is provided.

        Examples:
            # Example of using the enhance method
            noisy_input = torch.randn(10, 100, 64)  # Example noisy spectrum
            enhanced_output = score_model.enhance(
                noisy_specturm=noisy_input,
                sampler_type="pc",
                predictor="reverse_diffusion",
                corrector="ald",
                N=30,
                corrector_steps=1,
                snr=0.5
            )
        """
        Y = noisy_specturm.permute(0, 2, 1).unsqueeze(1)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(
                predictor,
                corrector,
                Y,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                **kwargs
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))

        X_Hat, nfe = sampler()

        X_Hat = X_Hat.squeeze(1).permute(0, 2, 1)

        return X_Hat
