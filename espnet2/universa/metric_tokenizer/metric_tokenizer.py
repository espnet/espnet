import json
from abc import ABC, abstractmethod
from operator import index
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from typeguard import typechecked


class AbsMetricTokenizer(ABC):
    @abstractmethod
    def metric2token(
        self,
        metrics: Dict[str, Union[float, str, int, Tuple[int, int]]],
        reduce_offset: bool = False,
    ) -> Dict[str, Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def token2metric(
        self, token: int, metric: Optional[str] = None
    ) -> Union[float, str]:
        raise NotImplementedError

    @abstractmethod
    def tokenseq2metric(
        self, tokens: Iterable[int], return_dict: bool = False
    ) -> Union[str, Dict[str, List[Union[float, int, str, Tuple[float, float]]]]]:
        raise NotImplementedError


class MetricTokenizer(AbsMetricTokenizer):
    def __init__(
        self,
        tokenizer_config: Union[str, Dict[str, Any]],
        tokenize_metric: Optional[List[str]],
    ):
        """Initialize the MetricTokenizer with a configuration file path.

        Args:
            tokenizer_config: The tokenizer configuration JSON file/Dictionary
            tokenize_metric: Selected known metric names, or None for all metrics
        """
        if isinstance(tokenizer_config, str):
            with open(tokenizer_config, "r") as f:
                config = json.load(f)
        else:
            config = tokenizer_config

        self.tokenizer_config = config["tokenizer"]
        self.vocab = config["VOCAB"]
        self.metric_offset = config["offset"]
        self.tokenize_metric = tokenize_metric
        if tokenize_metric is not None:
            unknown = set(tokenize_metric) - self.tokenizer_config.keys()
            if unknown:
                raise ValueError(f"Unknown selected metrics: {sorted(unknown)}")

        # Build inverse mapping for faster lookups
        self.overall_offset = 4  # Offset for the vocab indices
        self.vocab_indices = {
            token: idx + self.overall_offset for idx, token in enumerate(self.vocab)
        }
        self.vocab_indices["<pad>"] = 0  # Add padding token index
        self.vocab_indices["<unk>"] = 1  # Add unknown token index
        self.vocab_indices["<eos>"] = 2  # Add end-of-sequence token index
        self.vocab_indices["<sos>"] = 3  # Add start-of-sequence token index
        self.adjusted_vocab = ["<pad>", "<unk>", "<eos>", "<sos>"] + self.vocab

    def get_metric_meta_label(self, metric_name: str) -> int:
        """Get the meta label index for a given metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Index of the meta label in the vocabulary
        """
        meta_label_token = f"{metric_name}@meta_label"
        meta_label_index = self.vocab_indices.get(meta_label_token)

        if meta_label_index is None:
            raise ValueError(f"Invalid token: {meta_label_token}")

        return meta_label_index

    def get_token_list(self) -> List[str]:
        """Get the list of tokens in the vocabulary.

        Returns:
            List of tokens
        """
        return self.adjusted_vocab

    def get_token_index(
        self, metric_name: str, value_index: int, reduce_offset: bool = False
    ) -> Tuple[int, int]:
        """Get the token indices for a metric and its value.

        Args:
            metric_name: Name of the metric
            value_index: Index of the value for this metric
            reduce_offset: Whether to reduce metric-related offset

        Returns:
            Tuple of (meta_label_index, value_index) in the vocabulary
        """
        meta_label_token = f"{metric_name}@meta_label"
        value_token = f"{metric_name}@{value_index}"

        meta_label_index = self.vocab_indices.get(meta_label_token)
        value_index = self.vocab_indices.get(value_token)

        if meta_label_index is None or value_index is None:
            raise ValueError(f"Invalid token: {meta_label_token} or {value_token}")

        if reduce_offset:
            metric_offset, _ = self.metric_offset[metric_name]
            value_index = value_index - metric_offset - self.overall_offset

        return meta_label_index, value_index

    def _get_value_index(self, metric_name: str, value: Union[float, str, int]) -> int:
        """Determine the appropriate token index for a given metric value.

        Args:
            metric_name: Name of the metric
            value: The actual value of the metric

        Returns:
            Index of the appropriate value token
        """
        if isinstance(self.tokenizer_config[metric_name][0], str):
            # For categorical values, find its index in the list
            try:
                return self.tokenizer_config[metric_name].index(value) + 1
            except ValueError:
                raise ValueError(
                    f"Invalid category value: {value} for metric {metric_name}"
                )
        else:
            # For numerical metrics with thresholds
            thresholds = self.tokenizer_config[metric_name]
            # Find the first threshold that the value is less than
            for i, threshold in enumerate(thresholds):
                if value < threshold:
                    return i
            # If value is greater than all thresholds, return the last index
            return len(thresholds) - 1

    @typechecked
    def metric2token(
        self,
        metrics: Dict[str, Union[float, str, int, Tuple[int, int]]],
        reduce_offset: bool = False,
    ) -> Dict[str, Tuple[int, int]]:
        """Convert metrics dictionary to token indices.

        Args:
            metrics: Dictionary of metric names and their values. Unknown names
                raise ValueError; known metrics outside tokenize_metric are skipped.
            reduce_offset: Whether to reduce metric-related offset

        Returns:
            Dictionary of token indices for each metric (meta_label, value)
        """
        token_indices = {}

        for metric_name, value in metrics.items():
            if metric_name not in self.tokenizer_config:
                raise ValueError(f"Unknown metric: {metric_name}")
            if (
                self.tokenize_metric is not None
                and metric_name not in self.tokenize_metric
            ):
                continue

            if isinstance(value, tuple):
                # already tokenized
                token_indices[metric_name] = value
                continue

            # Get the index of the value in the corresponding metric's range
            value_index = self._get_value_index(metric_name, value)

            # Get the token indices for this metric and value
            token_pair = self.get_token_index(
                metric_name, value_index, reduce_offset=reduce_offset
            )
            token_indices[metric_name] = token_pair

        return token_indices

    @typechecked
    def tokenseq2metric(
        self, tokens: Iterable[int], return_dict: bool = False
    ) -> Union[str, Dict[str, List[Union[float, int, str, Tuple[float, float]]]]]:
        """Convert token indices back to a metric representation.

        Args:
            tokens: Iterable of token indices
            return_dict: If True, return a dictionary instead of a string

        Returns:
            String representation of the metrics
        """
        tokens_list = [index(token) for token in tokens]
        if not tokens_list:
            raise ValueError("Token list is empty")
        if tokens_list[0] == 2:
            # Published ARECHO checkpoints use ID 2 as the start token.
            tokens_list = tokens_list[1:]
        if len(tokens_list) % 2 != 0:
            raise ValueError(
                "Expected an even number of tokens (meta_label, value pairs)"
            )

        result = {}
        for i in range(0, len(tokens_list), 2):
            meta_label_idx = tokens_list[i]
            value_idx = tokens_list[i + 1]

            if not 0 <= meta_label_idx < len(self.adjusted_vocab):
                raise ValueError(f"Invalid token index: {meta_label_idx}")
            meta_label = self.adjusted_vocab[meta_label_idx]
            if not meta_label.endswith("@meta_label"):
                raise ValueError(f"Invalid meta_label: {meta_label}")
            metric_name = meta_label.removesuffix("@meta_label")
            result[metric_name] = [self.token2metric(value_idx, metric_name)]

        if return_dict:
            return result
        # Format the result as a string
        formatted_result = ", ".join([f"{k}: {v}" for k, v in result.items()])
        return formatted_result

    @typechecked
    def token2metric(
        self, token: int, metric: Optional[str] = None
    ) -> Union[float, str]:
        """Convert a single token index back to its metric representation.

        Args:
            token: Token index
            metric: Optional metric name for non-meta_label tokens

        Returns:
            String representation of the metric
        """
        if not 0 <= token < len(self.adjusted_vocab):
            raise ValueError(f"Invalid token index: {token}")

        token_result = self.adjusted_vocab[token]
        if metric is None:
            if token_result.endswith("@meta_label"):
                return token_result.removesuffix("@meta_label")
            raise ValueError("Metric name must be provided for non-meta_label tokens")
        if metric not in self.tokenizer_config:
            raise ValueError(f"Unknown metric: {metric}")
        prefix = f"{metric}@"
        suffix = token_result.removeprefix(prefix)
        if not token_result.startswith(prefix) or not suffix.isdecimal():
            raise ValueError(f"Invalid value token {token_result} for metric {metric}")
        token_value = int(suffix)
        values = self.tokenizer_config[metric]
        minimum = 1 if isinstance(values[0], str) else 0
        if not minimum <= token_value <= len(values):
            raise ValueError(f"Invalid token value: {token_value} for metric {metric}")
        # Preserve historical numeric bins, including the below-first-threshold bin.
        return values[max(token_value - 1, 0)]

    @typechecked
    def add_offset(self, src_tokens: Iterable[int], metric_name: str) -> List[int]:
        """Add back the offset related to metrics.

        Args:
            src_tokens: source tokens to be added
            metric_name: the name of the metric

        Returns:
            Token with added offset
        """
        offset = self.metric_offset[metric_name][0] + self.overall_offset
        return [int(t + offset) for t in src_tokens]
