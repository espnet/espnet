from abc import ABC, abstractmethod

import torch.optim.lr_scheduler as L


class AbsScheduler(ABC):
    """
        Abstract base class for defining learning rate schedulers.

    This class provides a blueprint for implementing various types of
    learning rate schedulers in PyTorch. Schedulers help to adjust the
    learning rate during training based on specific strategies.

    Classes inheriting from `AbsScheduler` should implement the following
    abstract methods:

    - `step`: Updates the learning rate based on the current epoch or
      other criteria.
    - `state_dict`: Returns the state of the scheduler as a dictionary.
    - `load_state_dict`: Loads the scheduler state from a given
      dictionary.

    Example usage:

    ```python
    class CustomScheduler(AbsScheduler):
        def __init__(self, optimizer):
            self.scheduler = L.StepLR(optimizer, step_size=10, gamma=0.1)

        def step(self, epoch: int = None):
            self.scheduler.step(epoch)

        def state_dict(self):
            return self.scheduler.state_dict()

        def load_state_dict(self, state):
            self.scheduler.load_state_dict(state)
    ```

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If any of the abstract methods are not
        implemented in a subclass.
    """

    @abstractmethod
    def step(self, epoch: int = None):
        """
                Abstract base class for defining custom learning rate schedulers.

        This class provides a blueprint for creating learning rate schedulers that can
        be integrated into training loops. It defines the required methods that any
        scheduler should implement, including the `step` method, which updates the
        learning rate based on the current epoch.

        Attributes:
            None

        Args:
            epoch (int, optional): The current epoch. If None, the method should handle
                it accordingly. Defaults to None.

        Returns:
            None

        Raises:
            NotImplementedError: If the derived class does not implement the `step`
                method.

        Examples:
            class CustomScheduler(AbsScheduler):
                def __init__(self, optimizer):
                    self.optimizer = optimizer

                def step(self, epoch=None):
                    # Custom logic to adjust the learning rate
                    pass

                def state_dict(self):
                    # Return the state of the scheduler
                    pass

                def load_state_dict(self, state):
                    # Load the state of the scheduler
                    pass

        Note:
            This is an abstract class and should not be instantiated directly.
            Derived classes must implement the abstract methods.

        Todo:
            Implement additional methods or attributes as needed for specific
            learning rate scheduling requirements.
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
                Abstract method to retrieve the state of the scheduler.

        This method should return a dictionary containing the current state of the
        scheduler. The returned state can be used to save the scheduler's state
        and later load it using the `load_state_dict` method.

        Returns:
            dict: A dictionary containing the state of the scheduler.

        Examples:
            # Example of how to use state_dict in a custom scheduler
            class CustomScheduler(AbsScheduler):
                def __init__(self):
                    self.state = {'epoch': 0, 'lr': 0.01}

                def step(self, epoch: int = None):
                    if epoch is not None:
                        self.state['epoch'] = epoch

                def state_dict(self):
                    return self.state

                def load_state_dict(self, state):
                    self.state = state

            scheduler = CustomScheduler()
            print(scheduler.state_dict())  # Output: {'epoch': 0, 'lr': 0.01}
        """
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """
                Loads the state of the learning rate scheduler from a given state dictionary.

        This method is intended to restore the internal state of the scheduler from a
        previously saved state. This is particularly useful when resuming training from
        a checkpoint, allowing the scheduler to continue its operation as expected.

        Args:
            state (dict): A state dictionary containing the parameters to load into the
            scheduler. This should be a dictionary that was previously saved by the
            `state_dict` method.

        Raises:
            ValueError: If the provided state does not match the expected format or
            contains invalid keys.

        Examples:
            # Example of saving and loading state in a custom scheduler
            scheduler = MyCustomScheduler()

            # Save the state
            state = scheduler.state_dict()

            # Load the state
            scheduler.load_state_dict(state)

        Note:
            This method should be implemented by subclasses to define how the state is
            restored, ensuring that all necessary attributes are correctly updated.

        Todo:
            Implement specific loading behavior in subclasses of this abstract method.
        """
        pass


# If you need to define custom scheduler, please inherit these classes
class AbsBatchStepScheduler(AbsScheduler):
    """
    Abstract base class for batch step learning rate schedulers.

    This class serves as a blueprint for creating custom learning rate
    schedulers that operate on a batch basis. It inherits from the
    `AbsScheduler` class and mandates the implementation of the
    `step`, `state_dict`, and `load_state_dict` methods.

    Attributes:
        None

    Args:
        epoch (int, optional): The current epoch number. Default is None.

    Returns:
        None

    Raises:
        NotImplementedError: If the method is not implemented in a
        subclass.

    Examples:
        To create a custom batch step scheduler, inherit from this
        class and implement the required methods:

        ```python
        class CustomBatchScheduler(AbsBatchStepScheduler):
            def step(self, epoch: int = None):
                # Custom implementation for step

            def state_dict(self):
                # Custom implementation to return state dict

            def load_state_dict(self, state):
                # Custom implementation to load state dict
        ```
    """

    @abstractmethod
    def step(self, epoch: int = None):
        """
                Abstract base class for batch step learning rate schedulers.

        This class defines the interface for batch step learning rate schedulers,
        which are used to adjust the learning rate based on the number of batches
        processed. Classes inheriting from this abstract class should implement
        the `step` method, which updates the learning rate according to the
        specific scheduling strategy.

        Attributes:
            None

        Args:
            epoch (int, optional): The current epoch number. If None, the epoch
                will not be used in the scheduling logic.

        Returns:
            None

        Raises:
            NotImplementedError: If the `step` method is not implemented in a
                subclass.

        Examples:
            To create a custom batch step scheduler, inherit from this class and
            implement the `step` method:

                class CustomBatchStepScheduler(AbsBatchStepScheduler):
                    def step(self, epoch: int = None):
                        # Implement your custom step logic here
                        pass

                    def state_dict(self):
                        # Return the state of the scheduler
                        pass

                    def load_state_dict(self, state):
                        # Load the state into the scheduler
                        pass
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
                Abstract base class for batch step learning rate schedulers.

        This class defines the interface for learning rate schedulers that operate on
        batches. Implementing classes should provide their own logic for adjusting
        the learning rate based on the batch steps.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            None

        Examples:
            To create a custom scheduler, inherit from AbsBatchStepScheduler and
            implement the required methods.

            class CustomBatchStepScheduler(AbsBatchStepScheduler):
                def step(self, epoch: int = None):
                    # Custom step logic
                    pass

                def state_dict(self):
                    # Return the state of the scheduler
                    return {}

                def load_state_dict(self, state):
                    # Load the state into the scheduler
                    pass
        """
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """
                Loads the state dictionary into the scheduler.

        This method is intended to restore the state of the scheduler from a previously
        saved state dictionary. It allows for resuming training from a specific point
        while maintaining the learning rate schedule.

        Args:
            state (dict): A dictionary containing the state of the scheduler, which
            typically includes the learning rate and any other relevant parameters
            that need to be restored.

        Raises:
            ValueError: If the state dictionary is invalid or does not match the
            expected structure.

        Examples:
            # Create a scheduler instance
            scheduler = MyCustomScheduler()

            # Load a previously saved state
            state_dict = torch.load('scheduler_state.pth')
            scheduler.load_state_dict(state_dict)
        """
        pass


class AbsEpochStepScheduler(AbsScheduler):
    """
        Abstract base class for epoch-based learning rate schedulers.

    This class defines the interface for all epoch-based schedulers. It inherits from
    the `AbsScheduler` class and requires implementations for methods to manage the
    learning rate at each epoch.

    Attributes:
        None

    Args:
        epoch (int, optional): The current epoch. Defaults to None.

    Methods:
        step(epoch: int = None):
            Updates the learning rate based on the current epoch.

        state_dict():
            Returns the state of the scheduler as a dictionary.

        load_state_dict(state):
            Loads the state of the scheduler from a given dictionary.

    Examples:
        # Example of a custom epoch scheduler implementation
        class CustomEpochScheduler(AbsEpochStepScheduler):
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.initial_lr = optimizer.param_groups[0]['lr']

            def step(self, epoch: int = None):
                new_lr = self.initial_lr * (0.1 ** (epoch // 10))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            def state_dict(self):
                return {'initial_lr': self.initial_lr}

            def load_state_dict(self, state):
                self.initial_lr = state['initial_lr']
    """

    @abstractmethod
    def step(self, epoch: int = None):
        """
            Abstract base class for epoch-based learning rate schedulers.

        This class serves as a blueprint for creating custom epoch-based learning
        rate schedulers by inheriting from it. Implementations must define the
        `step`, `state_dict`, and `load_state_dict` methods.

        Attributes:
            None

        Args:
            epoch (int, optional): The current epoch number. Defaults to None.

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the step method is not implemented by a subclass.

        Examples:
            To create a custom scheduler, you would inherit from this class and
            implement the required methods. For example:

            class CustomEpochScheduler(AbsEpochStepScheduler):
                def step(self, epoch: int = None):
                    # Custom logic to update learning rate
                    pass

                def state_dict(self):
                    # Logic to return the state dictionary
                    pass

                def load_state_dict(self, state):
                    # Logic to load the state dictionary
                    pass
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
            Abstract base class for epoch-based step learning rate schedulers.

        This class provides the interface for implementing learning rate
        schedulers that adjust the learning rate at the end of each epoch.

        Attributes:
            None

        Args:
            epoch (int, optional): The current epoch. Defaults to None.

        Returns:
            dict: A state dictionary containing the current state of the
            scheduler.

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not implemented in a
            subclass.

        Examples:
            To create a custom epoch step scheduler, inherit from this class
            and implement the required methods:

            ```python
            class MyCustomScheduler(AbsEpochStepScheduler):
                def step(self, epoch: int = None):
                    # Custom logic to update the learning rate
                    pass

                def state_dict(self):
                    # Return the state of the scheduler
                    return {}

                def load_state_dict(self, state):
                    # Load the state into the scheduler
                    pass
            ```
        """
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """
            Abstract base class for epoch step learning rate schedulers.

        This class defines the interface for epoch-based learning rate schedulers.
        Subclasses must implement the `step`, `state_dict`, and `load_state_dict`
        methods to manage the learning rate scheduling.

        Attributes:
            None

        Args:
            state: The state dictionary to load into the scheduler.

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyScheduler(AbsEpochStepScheduler):
                def step(self, epoch: int = None):
                    # Implementation for stepping the scheduler
                    pass

                def state_dict(self):
                    # Implementation for saving the state
                    pass

                def load_state_dict(self, state):
                    # Implementation for loading the state
                    pass

            my_scheduler = MyScheduler()
            my_scheduler.load_state_dict({'some_key': 'some_value'})

        Note:
            This class is meant to be subclassed and should not be instantiated
            directly.

        Todo:
            Implement the load_state_dict method in subclasses to handle
            specific state loading requirements.
        """
        pass


class AbsValEpochStepScheduler(AbsEpochStepScheduler):
    """
        Abstract base class for validation epoch step learning rate schedulers.

    This class serves as a blueprint for implementing custom learning rate
    schedulers that adjust the learning rate based on validation metrics
    and epoch count. Subclasses must implement the `step`, `state_dict`,
    and `load_state_dict` methods.

    Attributes:
        None

    Args:
        val (float): The validation metric to consider for adjusting the
            learning rate.
        epoch (int, optional): The current epoch number. If not provided,
            defaults to None.

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.

    Examples:
        class CustomValScheduler(AbsValEpochStepScheduler):
            def step(self, val, epoch=None):
                # Implement custom logic to adjust learning rate based on
                # validation metric and epoch.
                pass

            def state_dict(self):
                # Return the state of the scheduler.
                pass

            def load_state_dict(self, state):
                # Load the state of the scheduler.
                pass
    """

    @abstractmethod
    def step(self, val, epoch: int = None):
        """
                Abstract base class for defining a validation epoch step scheduler.

        This class serves as a blueprint for custom validation epoch step schedulers
        that can adjust learning rates based on validation metrics. Implementing
        classes must define the `step`, `state_dict`, and `load_state_dict` methods.

        Attributes:
            None

        Args:
            val (float): The validation metric used to determine the adjustment of the
                         learning rate.
            epoch (int, optional): The current epoch number. If not provided, defaults
                                   to None.

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is called without being overridden in
                                 a subclass.

        Examples:
            class CustomScheduler(AbsValEpochStepScheduler):
                def step(self, val, epoch=None):
                    # Implement logic to adjust learning rate based on val
                    pass

                def state_dict(self):
                    # Implement logic to return the state of the scheduler
                    pass

                def load_state_dict(self, state):
                    # Implement logic to load the state of the scheduler
                    pass
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
            Abstract base class for validation-based epoch step learning rate schedulers.

        This class defines the interface for creating custom learning rate schedulers
        that adjust the learning rate based on validation metrics at the end of each
        epoch.

        Attributes:
            None

        Args:
            val (float): The validation metric to base the learning rate adjustment on.
            epoch (int, optional): The current epoch number. Defaults to None.

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not overridden in a derived class.

        Examples:
            To create a custom scheduler, inherit from this class and implement the
            required methods:

            ```python
            class MyCustomScheduler(AbsValEpochStepScheduler):
                def step(self, val, epoch=None):
                    # Custom implementation for step
                    pass

                def state_dict(self):
                    # Return the state of the scheduler
                    pass

                def load_state_dict(self, state):
                    # Load the state into the scheduler
                    pass
            ```
        """
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """
            Abstract base class for value-based epoch step learning rate schedulers.

        This class extends the `AbsEpochStepScheduler` and provides a blueprint for
        implementing schedulers that adjust the learning rate based on validation
        metrics.

        Attributes:
            None

        Args:
            state (dict): A dictionary containing the state information to load.

        Raises:
            NotImplementedError: If the method is not overridden in a derived class.

        Examples:
            To create a custom scheduler, inherit from this class and implement
            the abstract methods.

            ```python
            class MyCustomScheduler(AbsValEpochStepScheduler):
                def step(self, val, epoch=None):
                    # Custom logic for adjusting the learning rate based on val
                    pass

                def state_dict(self):
                    # Return the state of the scheduler
                    pass

                def load_state_dict(self, state):
                    # Load the state into the scheduler
                    pass
            ```

        Note:
            This class is designed to be inherited. Ensure that all abstract methods
            are implemented in the derived class.

        Todo:
            Implement specific learning rate scheduling strategies based on validation
            metrics in derived classes.
        """
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class
# to judge these classes.
AbsValEpochStepScheduler.register(L.ReduceLROnPlateau)
for s in [
    L.ReduceLROnPlateau,
    L.LambdaLR,
    L.StepLR,
    L.MultiStepLR,
    L.MultiStepLR,
    L.ExponentialLR,
    L.CosineAnnealingLR,
]:
    AbsEpochStepScheduler.register(s)

AbsBatchStepScheduler.register(L.CyclicLR)
for s in [
    L.OneCycleLR,
    L.CosineAnnealingWarmRestarts,
]:
    AbsBatchStepScheduler.register(s)
