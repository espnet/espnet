"""Extract language embeddings of a manifest and plot them with t-SNE."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from egs3.voxlingua107.esp2_lid.src.prepared_dataset import Dataset


def extract_embeddings(model, dataset, max_utt_per_lang=1000, batch_size=4):
    """Return the embeddings of each reference language, in manifest order.

    Args:
        model: ``Speech2Language`` built with ``extract_embd=True``.
        dataset: Dataset whose ``examples`` hold the reference languages.
        max_utt_per_lang: Keep at most this many utterances per language, as
            ESPnet2 limits t-SNE inputs. Other utterances are not read.
        batch_size: Utterances per padded forward pass.

    Returns:
        A dictionary from language code to a list of L2-normalized embeddings.

    Example:
        >>> languages = extract_embeddings(model, Dataset(manifest, lang2utt))
    """
    taken = Counter()
    selected = []
    for idx, example in enumerate(dataset.examples):
        if taken[example.language] < max_utt_per_lang:
            taken[example.language] += 1
            selected.append(idx)

    languages = defaultdict(list)
    for start in range(0, len(selected), batch_size):
        items = [dataset[idx] for idx in selected[start : start + batch_size]]
        predictions = model([item["speech"] for item in items])
        for item, prediction in zip(items, predictions):
            languages[item["lid_labels"]].append(prediction["embedding"])
    return dict(languages)


def write_embeddings(languages, output_dir, tsne=True, seed=0):
    """Write embedding archives, language averages and t-SNE plots.

    Writes ``lang_to_embds.npz``, ``lang_to_avg_embd.npz`` with L2-normalized
    language averages, and ``tsne_plots/`` from ESPnet2's plotting helper,
    under ``output_dir``. Existing files are replaced.

    Args:
        languages: Language code to list of embeddings.
        output_dir: Destination directory, created when missing.
        tsne: Also plot the embeddings and the averages.
        seed: Random seed for t-SNE.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    averages = {
        language: torch.nn.functional.normalize(
            torch.from_numpy(np.mean(embeddings, axis=0)), p=2, dim=0
        ).numpy()
        for language, embeddings in languages.items()
    }
    np.savez(output_dir / "lang_to_embds.npz", **languages)
    np.savez(output_dir / "lang_to_avg_embd.npz", **averages)
    if not tsne:
        return

    from espnet2.bin.lid_inference import gen_tsne_plot

    count = sum(len(embeddings) for embeddings in languages.values())
    for embeddings, num_points in ((languages, count), (averages, len(averages))):
        if num_points > 1:
            # t-SNE needs a perplexity below the number of points.
            gen_tsne_plot(
                embeddings,
                str(output_dir / "tsne_plots"),
                seed,
                perplexity=min(5, num_points - 1),
            )


def main():
    """Extract the embeddings of one manifest with a trained LID model."""
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--lid-train-config", required=True)
    argparser.add_argument("--lid-model-file", required=True)
    argparser.add_argument("--lang2utt", required=True)
    argparser.add_argument("--manifest", required=True)
    argparser.add_argument("--output-dir", type=Path, required=True)
    argparser.add_argument("--max-utt-per-lang", type=int, default=1000)
    argparser.add_argument("--batch-size", type=int, default=4)
    argparser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    argparser.add_argument("--no-tsne", action="store_true")
    argparser.add_argument("--seed", type=int, default=0)
    args = argparser.parse_args()

    from espnet3.systems.esp2_lid.src.speech2language import Speech2Language

    model = Speech2Language(
        args.lid_train_config,
        args.lid_model_file,
        args.lang2utt,
        device=args.device,
        extract_embd=True,
    )
    dataset = Dataset(args.manifest, lang2utt=args.lang2utt)
    languages = extract_embeddings(
        model, dataset, args.max_utt_per_lang, args.batch_size
    )
    write_embeddings(languages, args.output_dir, tsne=not args.no_tsne, seed=args.seed)


if __name__ == "__main__":
    main()
