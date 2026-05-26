"""Tests for parquet to jsonl preprocessing."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from tools.preprocess_parquet_to_jsonl import parquet_to_jsonl


def test_parquet_to_jsonl_filters_empty_text_and_writes_json(tmp_path):
    parquet_path = tmp_path / "part_000.parquet"
    table = pa.table({"text": [" hello ", "", None, "world"], "other": [1, 2, 3, 4]})
    pq.write_table(table, parquet_path)

    output = tmp_path / "out"
    parquet_to_jsonl(
        parquet_files=[parquet_path],
        output_dir=output,
        file_prefix="nemotron",
        text_column="text",
        batch_size=2,
        max_rows_per_file=10,
        output_path=None,
    )

    files = sorted(output.glob("nemotron_part_*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert rows == [{"text": "hello"}, {"text": "world"}]
