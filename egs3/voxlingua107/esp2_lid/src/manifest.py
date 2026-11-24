"""Parallel audio preparation with one manifest fragment per worker."""

from pathlib import Path

from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files


class ManifestRunner(BaseRunner):
    """Cut/resample file-backed utterances and concatenate their manifest rows.

    Args:
        provider: DownloadProvider containing destination/example dictionaries.
        output_dir: Split directory containing the final manifest and audio.

    Example:
        >>> ManifestRunner(provider, output_dir="data/voxpopuli/test",
        ...                shard_subdir=".prepare", resume=False)(range(len(tasks)))
    """

    @staticmethod
    def forward(index, dataset, **env):
        """Write one utterance to its ID-derived path and return its TSV row."""
        from egs3.voxlingua107.esp2_lid.src.prepare_utils import _write_example

        task = dataset[index]
        return _write_example(Path(task["destination"]), task["example"])

    @staticmethod
    def open_writers(shard_dir, **env):
        """Open a manifest fragment owned by one worker."""
        return {"manifest": (shard_dir / "manifest.tsv").open("w", encoding="utf-8")}

    @staticmethod
    def write_record(writers, result, state, **env):
        """Stream rows instead of retaining all processed audio in runner state."""
        writers["manifest"].write(result)

    def merge(self, shard_dirs):
        """Publish a manifest only after all audio-preparation shards finish."""
        if not shard_dirs:
            (self.output_dir / "manifest.tsv").write_text("", encoding="utf-8")
            return 0
        temporary = self.output_dir / "manifest.tsv.tmp"
        concatenate_shard_files(shard_dirs, "manifest.tsv", temporary)
        temporary.replace(self.output_dir / "manifest.tsv")
        with (self.output_dir / "manifest.tsv").open(encoding="utf-8") as stream:
            return sum(1 for _ in stream)
