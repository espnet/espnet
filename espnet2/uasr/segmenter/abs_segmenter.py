"""
Segmenter definition for UASR task

Practially, the output of the generator (in frame-level) may
predict the same phoneme for consecutive frames, which makes
it too easy for the discriminator. So, the segmenter here is
to merge frames with a similar prediction from the generator output.
"""

from abc import ABC, abstractmethod

import torch


class AbsSegmenter(torch.nn.Module, ABC):
    """
        Segmenter definition for UASR (Unsupervised Automatic Speech Recognition) task.

    This abstract base class provides the structure for segmenting frame-level
    outputs from a generator. In practice, consecutive frames may predict the
    same phoneme, making it too easy for the discriminator. Therefore, this
    segmenter is designed to merge frames with similar predictions from the
    generator output.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the abstract methods are not implemented by
        subclasses.

    Examples:
        class MySegmenter(AbsSegmenter):
            def pre_segment(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
                # Implementation of pre_segment
                pass

            def logit_segment(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
                # Implementation of logit_segment
                pass

    Note:
        This class is meant to be subclassed. The methods `pre_segment` and
        `logit_segment` must be implemented by any subclass.

    Todo:
        Implement concrete segmenter classes that inherit from AbsSegmenter.
    """

    @abstractmethod
    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        """
                Segmenter definition for UASR task.

        Practically, the output of the generator (in frame-level) may
        predict the same phoneme for consecutive frames, which makes
        it too easy for the discriminator. So, the segmenter here is
        to merge frames with a similar prediction from the generator output.

        Methods:
            pre_segment(xs_pad: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
                Abstract method to perform pre-segmentation on the input data.

            logit_segment(xs_pad: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
                Abstract method to obtain logits for segmentation from the input data.

        Attributes:
            None

        Args:
            xs_pad (torch.Tensor): A tensor containing padded input sequences.
            ilens (torch.Tensor): A tensor containing the lengths of the input sequences.

        Returns:
            torch.Tensor: A tensor containing the segmented output.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            To use this class, you must inherit from it and implement the abstract methods:

            class MySegmenter(AbsSegmenter):
                def pre_segment(self, xs_pad, ilens):
                    # Implementation of pre-segmentation logic
                    pass

                def logit_segment(self, xs_pad, ilens):
                    # Implementation of segmentation logits logic
                    pass
        """
        raise NotImplementedError

    @abstractmethod
    def logit_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        """
                Segmenter definition for UASR task.

        This class provides an abstract base for segmenting audio frames based on the
        output of a generator in the UASR (Unsupervised Automatic Speech Recognition)
        task. The segmenter is designed to merge frames with similar predictions,
        thereby improving the discriminator's ability to learn from the generator's
        output.

        Attributes:
            None

        Args:
            xs_pad (torch.Tensor): A padded tensor containing the input audio frames.
            ilens (torch.Tensor): A tensor containing the lengths of the input
                sequences.

        Returns:
            torch.Tensor: A tensor containing the segmented audio frames.

        Raises:
            NotImplementedError: If the method is called directly from this abstract
                class without being overridden in a derived class.

        Examples:
            class MySegmenter(AbsSegmenter):
                def pre_segment(self, xs_pad, ilens):
                    # Implementation of pre-segment method
                    pass

                def logit_segment(self, xs_pad, ilens):
                    # Implementation of logit_segment method
                    return segmented_output

            segmenter = MySegmenter()
            output = segmenter.logit_segment(xs_pad, ilens)

        Note:
            This class is intended to be subclassed. Implementations of the abstract
            methods must be provided in derived classes.
        """
        raise NotImplementedError
