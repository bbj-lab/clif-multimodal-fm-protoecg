"""Streaming Parquet writer for large sequence outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class StreamingParquetWriter:
    """Buffered Parquet writer that flushes row groups periodically.

    Avoids holding all sequences in memory at once.
    """

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        buffer_size: int = 5000,
        compression: str = "zstd",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.buffer_size = buffer_size
        self.compression = compression
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._total_rows = 0

    def write_row(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self._flush()

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        self._buffer.extend(rows)
        if len(self._buffer) >= self.buffer_size:
            self._flush()

    def close(self) -> int:
        """Flush remaining buffer and close. Returns total rows written."""
        if self._buffer:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._total_rows

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self.path), self.schema, compression=self.compression
            )
        self._writer.write_table(table)
        self._total_rows += len(self._buffer)
        self._buffer.clear()

    def __enter__(self) -> StreamingParquetWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
