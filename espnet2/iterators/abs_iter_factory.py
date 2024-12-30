from abc import ABC, abstractmethod
from typing import Iterator


class AbsIterFactory(ABC):
    """
        Abstract base class for iterator factories in the ESPnet2 framework.

    This class defines a blueprint for creating iterators, which can be used to
    iterate over datasets in a structured manner. Concrete implementations of this
    class must provide an implementation for the `build_iter` method, which is
    responsible for constructing the iterator based on the specified epoch and
    shuffle parameters.

    Attributes:
        None

    Args:
        epoch (int): The current epoch number. This can be used to control the
            behavior of the iterator (e.g., shuffling).
        shuffle (bool, optional): A flag indicating whether to shuffle the data
            before iteration. Defaults to None.

    Returns:
        Iterator: An iterator object that yields data items.

    Raises:
        NotImplementedError: If the `build_iter` method is not implemented by a
            subclass.

    Examples:
        To create a custom iterator factory, subclass `AbsIterFactory` and
        implement the `build_iter` method:

        ```python
        class CustomIterFactory(AbsIterFactory):
            def build_iter(self, epoch: int, shuffle: bool = False) -> Iterator:
                # Custom iterator logic goes here
                ...
        ```

    Note:
        This class is part of the ESPnet2 library and is intended for internal
        use within the framework.
    """

    @abstractmethod
    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        """
            Builds an iterator for the specified epoch.

        This method is responsible for creating an iterator that can be used to
        traverse through the data for a given epoch. The iterator can be shuffled
        based on the provided parameter.

        Args:
            epoch (int): The epoch number for which the iterator is being built.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to None.

        Returns:
            Iterator: An iterator that allows iteration over the data for the
            specified epoch.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.

        Examples:
            class MyIterFactory(AbsIterFactory):
                def build_iter(self, epoch: int, shuffle: bool = False) -> Iterator:
                    # Implementation of iterator building logic here
                    pass

            factory = MyIterFactory()
            iterator = factory.build_iter(epoch=1, shuffle=True)
            for item in iterator:
                print(item)

        Note:
            The `shuffle` parameter's default value should be explicitly handled in
            the implementation to ensure consistent behavior.
        """
        raise NotImplementedError
