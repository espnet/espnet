from arkive import Arkive
import multiprocessing as mp
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


def dump_arkive_singleprocess(
    output_dir,
    data_name,
    data_dict,
    data_type="text",
    target_format="string",
):
    writer = Arkive(str(output_dir), data_name)
    writer.append(
        list(data_dict.values()),
        utt_ids=list(data_dict.keys()),
        data_type=data_type,
        compression_level=10,
        num_workers=0,
        target_format=target_format,
    )


class ArkiveWriter:
    """A writer that accumulates data and dumps in chunks using multiprocessing.

    When the accumulated data exceeds chunk_size, it dumps the chunks
    via multiprocessing. Residual data is kept for the next write call.
    """

    def __init__(
        self,
        output_dir,
        data_name,
        data_type="text",
        target_format="string",
        chunk_size=30000,
        max_workers=4,
    ):
        self.output_dir = Path(output_dir)
        self.data_name = data_name
        self.data_type = data_type
        self.target_format = target_format
        self.chunk_size = chunk_size
        self.max_workers = max_workers

        self._buffer = {}
        self._chunk_id = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, data_dict):
        """Write data to the arkive.

        Accumulates data and dumps chunks when buffer exceeds chunk_size.
        Residual data is kept for the next write call.

        Args:
            data_dict: Dictionary mapping keys to values.
        """
        self._buffer.update(data_dict)

        if len(self._buffer) < self.chunk_size:
            return

        keys = list(self._buffer.keys())
        num_full_chunks = len(keys) // self.chunk_size
        num_to_dump = num_full_chunks * self.chunk_size

        keys_to_dump = keys[:num_to_dump]
        keys_to_keep = keys[num_to_dump:]

        chunks = [
            keys_to_dump[i : i + self.chunk_size]
            for i in range(0, len(keys_to_dump), self.chunk_size)
        ]

        args_list = []
        for chunk_keys in chunks:
            self._chunk_id += 1
            chunk_dict = {k: self._buffer[k] for k in chunk_keys}
            chunk_output_dir = self.output_dir / f"split_{self._chunk_id}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)
            args_list.append(
                (
                    chunk_output_dir,
                    self.data_name,
                    chunk_dict,
                    self.data_type,
                    self.target_format,
                )
            )

        # Clean up buffer before multiprocessing to reduce memory usage
        self._buffer = {k: self._buffer[k] for k in keys_to_keep}

        print("start multiprocessing writing...", flush=True)
        with mp.Pool(processes=self.max_workers) as pool:
            pool.starmap(dump_arkive_singleprocess, args_list)

    def finalize(self):
        """Flush any remaining data in the buffer and merge metadata."""
        if self._buffer:
            self._chunk_id += 1
            chunk_output_dir = self.output_dir / f"split_{self._chunk_id}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            dump_arkive_singleprocess(
                chunk_output_dir,
                self.data_name,
                self._buffer,
                self.data_type,
                self.target_format,
            )
            self._buffer = {}

        # Merge all metadata.parquet files from split directories
        self._merge_metadata()

    def _merge_metadata(self):
        """Merge all metadata.parquet files from split directories (streaming)."""
        parquet_files = []
        for chunk_id in range(1, self._chunk_id + 1):
            parquet_path = self.output_dir / f"split_{chunk_id}" / "metadata.parquet"
            if parquet_path.exists():
                parquet_files.append(parquet_path)

        if not parquet_files:
            return

        merged_path = self.output_dir / "metadata.parquet"
        writer = None

        for parquet_file in parquet_files:
            parquet_reader = pq.ParquetFile(str(parquet_file))
            for batch in parquet_reader.iter_batches():
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(str(merged_path), table.schema)
                writer.write_table(table)

        if writer is not None:
            writer.close()
