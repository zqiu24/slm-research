"""Convert parquet text shards to Megatron-compatible JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


class RotatingJsonlWriter:
    def __init__(self, output_dir: Path, file_prefix: str, max_rows_per_file: int):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.max_rows = max_rows_per_file
        self.current_file_idx = 0
        self.current_file_rows = 0
        self.file_handle = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_next_file()

    def _open_next_file(self) -> None:
        if self.file_handle:
            self.file_handle.close()
        path = self.output_dir / f"{self.file_prefix}_part_{self.current_file_idx:05d}.jsonl"
        print(f"creating {path}")
        self.file_handle = path.open("w", encoding="utf-8")
        self.current_file_idx += 1
        self.current_file_rows = 0

    def write(self, row: dict[str, str]) -> None:
        self.file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.current_file_rows += 1
        if self.current_file_rows >= self.max_rows:
            self._open_next_file()

    def close(self) -> None:
        if self.file_handle:
            self.file_handle.close()


def parquet_to_jsonl(
    *,
    parquet_files: list[Path],
    output_dir: Path,
    file_prefix: str = "nemotron",
    text_column: str = "text",
    batch_size: int = 4096,
    max_rows_per_file: int = 100_000,
    output_path: Path | None = None,
) -> int:
    total_docs = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    handle = None
    if output_path is None:
        writer = RotatingJsonlWriter(output_dir, file_prefix, max_rows_per_file)
    else:
        handle = output_path.open("w", encoding="utf-8")

    try:
        for parquet_file in sorted(Path(p) for p in parquet_files):
            parquet = pq.ParquetFile(parquet_file)
            for batch in tqdm(
                parquet.iter_batches(batch_size=batch_size, columns=[text_column]),
                desc=f"reading {parquet_file.name}",
                total=parquet.num_row_groups,
            ):
                table = batch.to_pydict()
                for raw_text in table.get(text_column, []):
                    if raw_text is None:
                        continue
                    text = str(raw_text).strip()
                    if not text:
                        continue
                    row = {"text": text}
                    if handle is not None:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        writer.write(row)
                    total_docs += 1
    finally:
        if handle is not None:
            handle.close()
        if writer is not None:
            writer.close()
    print(f"processed_documents={total_docs}")
    return total_docs


def _resolve_input(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise RuntimeError(f"No parquet files found under {path}")
        return files
    if path.is_file():
        return [path]
    raise RuntimeError(f"Input path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="nemotron")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    files = _resolve_input(Path(args.input))
    output_path = None
    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(files):
            raise RuntimeError(f"Index {args.idx} out of range [0, {len(files) - 1}]")
        selected = files[args.idx]
        files = [selected]
        output_path = Path(args.output_dir) / f"{args.prefix}_{selected.stem}.jsonl"

    parquet_to_jsonl(
        parquet_files=files,
        output_dir=Path(args.output_dir),
        file_prefix=args.prefix,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_rows_per_file=args.max_rows,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
