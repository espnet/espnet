"""Language embedding summaries and ESPnet2 t-SNE visualization."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from espnet3.components.metrics.base_metric import BaseMetric


class Embedding(BaseMetric):
    """Summarize optional inference embeddings and reuse ESPnet2 t-SNE plots.

    Args:
        ref_key: SCP input alias holding reference language codes. Use predicted
            codes instead by setting the metric's ``inputs.ref: hyp``.
        embedding_key: SCP input alias holding per-utterance NumPy file paths.
        max_utt_per_lang: Keep at most this many embeddings per language, in SCP
            order, matching the ESPnet2 limit. Summaries use this subset too.
        save_tsne_plot: Also call ESPnet2's plotting helper. Its optional plotting
            dependencies are imported only when this option is true.
        seed: Random seed passed to the ESPnet2 t-SNE helper.
        perplexity: t-SNE perplexity, capped below the number of plotted points.
        max_iter: Number of t-SNE iterations (at least 250).

    Examples:
        Select this class in ``metrics`` with ``inputs: {ref: ref,
        embedding: embedding}``, after running inference with
        ``model.extract_embd: true`` and ``embedding`` in ``output_keys``.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        embedding_key: str = "embedding",
        max_utt_per_lang: int = 1000,
        save_tsne_plot: bool = False,
        seed: int = 0,
        perplexity: float = 5.0,
        max_iter: int = 1000,
    ) -> None:
        """Configure embedding summaries and optional t-SNE visualization."""
        if max_utt_per_lang < 1:
            raise ValueError("max_utt_per_lang must be positive")
        if perplexity <= 0 or max_iter < 250:
            raise ValueError("perplexity must be positive and max_iter at least 250")
        self.ref_key = ref_key
        self.embedding_key = embedding_key
        self.max_utt_per_lang = max_utt_per_lang
        self.save_tsne_plot = save_tsne_plot
        self.seed = seed
        self.perplexity = perplexity
        self.max_iter = max_iter

    def __call__(self, data: dict[str, Path], test_name: str, inference_dir: Path):
        """Write language embedding archives and optional plots for ``measure``.

        Args:
            data: Aligned reference and embedding SCP input paths.
            test_name: Inference test-set directory name.
            inference_dir: Root inference directory. Writes
                ``<test_name>_lang_to_embds.npz``, normalized language averages
                in ``<test_name>_lang_to_avg_embd.npz``, and optional
                ``tsne_plots/`` beneath its test-set directory. Re-running
                overwrites these derived artifacts, not the input embeddings.

        Returns:
            Counts of summarized embeddings and languages.

        Raises:
            ValueError: If inputs are empty, non-finite, or dimensionally invalid.
            AssertionError: If SCP utterance IDs or lengths do not match.
        """
        languages = defaultdict(list)
        dimension = None
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.embedding_key):
            language = row[self.ref_key].strip()
            if not language:
                raise ValueError(f"Empty LID label found for utterance: {utt_id}")
            if len(languages[language]) >= self.max_utt_per_lang:
                continue
            embedding = np.load(row[self.embedding_key], allow_pickle=False)
            if (
                embedding.ndim != 1
                or embedding.size < 2
                or not np.issubdtype(embedding.dtype, np.floating)
                or not np.isfinite(embedding).all()
            ):
                raise ValueError(f"Invalid language embedding for utterance: {utt_id}")
            if dimension is not None and embedding.size != dimension:
                raise ValueError(f"Inconsistent embedding dimensions: {utt_id}")
            dimension = embedding.size
            languages[language].append(embedding)
        if not languages:
            raise ValueError("No language embeddings found")

        averages = {
            language: torch.nn.functional.normalize(
                torch.from_numpy(np.mean(embeddings, axis=0)), p=2, dim=0
            ).numpy()
            for language, embeddings in languages.items()
        }
        output_dir = Path(inference_dir) / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"{test_name}_lang_to_embds.npz", **languages)
        np.savez(output_dir / f"{test_name}_lang_to_avg_embd.npz", **averages)

        count = sum(len(embeddings) for embeddings in languages.values())
        if self.save_tsne_plot:
            from espnet2.bin.lid_inference import gen_tsne_plot

            for embeddings, num_points in (
                (languages, count),
                (averages, len(averages)),
            ):
                if num_points > 1:
                    gen_tsne_plot(
                        embeddings,
                        str(output_dir / "tsne_plots"),
                        self.seed,
                        perplexity=min(self.perplexity, num_points - 1),
                        max_iter=self.max_iter,
                    )
        return {"Embeddings": count, "Languages": len(languages)}
