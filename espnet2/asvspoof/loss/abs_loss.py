from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsASVSpoofLoss(torch.nn.Module, ABC):
    """
    Base class for all ASV Spoofing loss modules.

    This abstract base class defines the interface for ASV Spoofing loss
    functions. Subclasses must implement the `forward` and `score`
    methods to compute the loss and score, respectively.

    Attributes:
        name (str): A string representing the name of the loss function,
        which will be used as a key in the reporter.

    Methods:
        forward(ref, inf) -> torch.Tensor:
            Computes the loss given reference and inferred values.

        score(pred) -> torch.Tensor:
            Computes the score based on the predictions.

    Raises:
        NotImplementedError: If the `forward` or `score` methods are not
        implemented in the subclass.

    Examples:
        To create a custom loss module, inherit from this class and
        implement the `forward` and `score` methods:

        ```python
        class CustomASVSpoofLoss(AbsASVSpoofLoss):
            @property
            def name(self) -> str:
                return "custom_loss"

            def forward(self, ref, inf) -> torch.Tensor:
                # Implement custom loss computation
                return loss

            def score(self, pred) -> torch.Tensor:
                # Implement scoring mechanism
                return score
        ```

    Note:
        Ensure that the returned tensor from `forward` is of shape (batch).
    """

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        """
        Base class for all ASV Spoofing loss modules.

        This abstract base class defines the structure and interface for ASV
        (Automatic Speaker Verification) spoofing loss functions. Subclasses
        should implement the `forward` and `score` methods to provide specific
        loss calculations and scoring mechanisms.

        Attributes:
            name (str): A string representing the name of the loss module.
                        Subclasses must implement this property.

        Methods:
            forward(ref, inf) -> torch.Tensor:
                Computes the loss given reference and input tensors.

            score(pred) -> torch.Tensor:
                Evaluates the predictions and returns a score tensor.

        Raises:
            NotImplementedError: If `name` property, `forward`, or `score`
                                methods are not implemented in a subclass.

        Examples:
            >>> class MyASVSpoofLoss(AbsASVSpoofLoss):
            ...     @property
            ...     def name(self):
            ...         return "MyASVSpoofLoss"
            ...
            ...     def forward(self, ref, inf):
            ...         # Implementation of loss calculation
            ...         return torch.tensor(0.0)  # Example output
            ...
            ...     def score(self, pred):
            ...         # Implementation of scoring
            ...         return torch.tensor(1.0)  # Example score

            >>> loss_module = MyASVSpoofLoss()
            >>> print(loss_module.name)
            MyASVSpoofLoss
        """
        return NotImplementedError

    @abstractmethod
    def forward(
        self,
        ref,
        inf,
    ) -> torch.Tensor:
        """
        Computes the forward pass of the ASV spoofing loss.

        This method calculates the loss based on the reference and
        inference tensors provided. It is expected that the input
        tensors have compatible shapes for loss computation.

        Args:
            ref (torch.Tensor): A tensor containing the reference values
                (ground truth) for the ASV spoofing task. The shape should
                be (batch_size, ...), where '...' can represent additional
                dimensions depending on the specific implementation.
            inf (torch.Tensor): A tensor containing the inference values
                predicted by the model. The shape should match the shape
                of the 'ref' tensor.

        Returns:
            torch.Tensor: A tensor containing the computed loss value,
            which should be of shape (batch_size,). The value represents
            the loss for each sample in the batch.

        Raises:
            NotImplementedError: If the method is not overridden in a
            subclass.

        Examples:
            >>> ref = torch.tensor([[0.0], [1.0]])
            >>> inf = torch.tensor([[0.1], [0.9]])
            >>> loss = model.forward(ref, inf)
            >>> print(loss)  # Example output: tensor([0.01, 0.01])

        Note:
            This is an abstract method and should be implemented in
            subclasses of `AbsASVSpoofLoss`.
        """
        # the return tensor should be shape of (batch)
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        pred,
    ) -> torch.Tensor:
        """
        Calculates the score based on the model's predictions.

        This method takes the model predictions as input and computes a score
        that reflects the performance of the ASV spoofing model. The specific
        scoring mechanism should be implemented in the derived classes.

        Args:
            pred (torch.Tensor): A tensor containing the model predictions.
                The shape of the tensor should be (batch_size, num_classes).

        Returns:
            torch.Tensor: A tensor containing the calculated scores, with a
            shape of (batch_size,).

        Raises:
            NotImplementedError: If the method is not implemented in a
            derived class.

        Examples:
            >>> model = MyASVSpoofLoss()  # Assume MyASVSpoofLoss implements this
            >>> predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
            >>> scores = model.score(predictions)
            >>> print(scores)  # Output will depend on the specific implementation

        Note:
            The score computation can vary significantly depending on the
            derived class and the specific ASV spoofing loss method being
            implemented.
        """
        raise NotImplementedError
