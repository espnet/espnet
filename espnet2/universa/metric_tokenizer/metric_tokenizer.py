import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from typeguard import typechecked


class AbsMetricTokenizer(ABC):
    @abstractmethod
    def metric2token(
        self, metrics: Dict[str, Union[float, str, int]]
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def token2metric(self, tokens: Iterable[int]) -> str:
        raise NotImplementedError


class MetricTokenizer(AbsMetricTokenizer):
    def __init__(
        self, tokenizer_config: Union[str, Dict[str, Any]], tokenize_metric: List[str]
    ):
        """
        Initialize the MetricTokenizer with a configuration file path.

        Args:
            tokenizer_config: The tokenizer configuration JSON file/Dictionary
            tokenize_metric: List of metric names to be tokenized
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

        # Extract metric types and their thresholds/categories
        self.metrics = {}
        for metric_name, values in self.tokenizer_config.items():
            self.metrics[metric_name] = values

    def get_metric_meta_label(self, metric_name: str) -> int:
        """
        Get the meta label index for a given metric.

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
        """
        Get the list of tokens in the vocabulary.

        Returns:
            List of tokens
        """
        return self.adjusted_vocab

    def get_token_index(
        self, metric_name: str, value_index: int, reduce_offset: bool = False
    ) -> Tuple[int, int]:
        """
        Get the token indices for a metric and its value.

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
        """
        Determine the appropriate token index for a given metric value.

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
        """
        Convert metrics dictionary to token indices.

        Args:
            metrics: Dictionary of metric names and their values
            return_dict: If True, return a dictionary of token indices instead of a list
            reduce_offset: Whether to reduce metric-related offset

        Returns:
            Dictionary of token indices for each metric (meta_label, value)
        """
        token_indices = {}

        for metric_name, value in metrics.items():
            if (
                self.tokenize_metric is not None
                and metric_name not in self.tokenize_metric
            ):
                continue

            if metric_name not in self.metrics:
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
        """
        Convert token indices back to a metric representation.

        Args:
            tokens: Iterable of token indices
            return_dict: If True, return a dictionary instead of a string

        Returns:
            String representation of the metrics
        """
        tokens_list = list(tokens)
        assert len(tokens_list) > 0, "Token list is empty"
        if tokens_list[0] == 2:
            # Remove the first token if it is <sos>
            tokens_list = tokens_list[1:]
        if len(tokens_list) % 2 != 0:
            raise ValueError(
                "Expected an even number of tokens (meta_label, value pairs)"
            )

        result = {}
        for i in range(0, len(tokens_list), 2):
            meta_label_idx = tokens_list[i]
            value_idx = tokens_list[i + 1]

            meta_label = self.adjusted_vocab[meta_label_idx]
            value_token = self.adjusted_vocab[value_idx]

            assert meta_label.endswith(
                "@meta_label"
            ), f"Invalid meta_label: {meta_label}"
            # Extract metric name from meta_label
            metric_name = meta_label.split("@meta_label")[0]

            # Extract value index from value token
            value_index = int(value_token.split("@")[-1])
            assert (
                value_index >= 0
            ), f"Invalid value index: {value_index} for token {value_token}"

            if metric_name not in self.tokenizer_config.keys():
                raise ValueError(f"Unknown metric in decoding: {metric_name}")

            # For category, get the actual category value
            if isinstance(self.tokenizer_config[metric_name][0], str):
                result[metric_name] = [
                    self.tokenizer_config[metric_name][value_index - 1]
                ]
            else:
                # For numerical metrics, represent as ranges
                thresholds = self.tokenizer_config[metric_name]
                if value_index == 0:
                    result[metric_name] = [thresholds[0]]
                elif value_index == len(thresholds):
                    result[metric_name] = [thresholds[-1]]
                else:
                    result[metric_name] = [thresholds[value_index - 1]]

        if return_dict:
            return result
        # Format the result as a string
        formatted_result = ", ".join([f"{k}: {v}" for k, v in result.items()])
        return formatted_result

    @typechecked
    def token2metric(
        self, token: int, metric: Optional[str] = None
    ) -> Union[float, str]:
        """
        Convert a single token index back to its metric representation.

        Args:
            token: Token index
            metric: Optional metric name for non-meta_label tokens

        Returns:
            String representation of the metric
        """
        if token not in self.vocab_indices.values():
            raise ValueError(f"Invalid token index: {token}")

        token_result = self.adjusted_vocab[token]
        if token_result.endswith("@meta_label"):
            metric_name = token_result.split("@meta_label")[0]
            return metric_name
        else:
            token_value = int(token_result.split("@")[-1])
            if metric is None:
                raise ValueError(
                    "Metric name must be provided for non-meta_label tokens"
                )
            if metric not in self.tokenizer_config.keys():
                raise ValueError(f"Unknown metric: {metric}")

            # NOTE(jiatong): the first token is for padding
            assert token_value >= 1 and token_value <= len(
                self.tokenizer_config[metric]
            ), f"Invalid token value: {token_value} for metric {metric}"
            return self.tokenizer_config[metric][token_value - 1]

    @typechecked
    def add_offset(self, src_tokens: Iterable[int], metric_name: str) -> List[int]:
        """
        Add back the offset related to metrics.

        Args:
            src_tokens: source tokens to be added
            metric_name: the name of the metric

        Returns:
            Token with added offset
        """
        offset = self.metric_offset[metric_name][0]
        # NOTE(jiatong): +1 for position of meta label
        return [int(t + offset) for t in src_tokens]
