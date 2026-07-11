"""Streaming CSV I/O + a minimal PyTorch IterableDataset for SMILES pairs.

All CSV reading goes through polars' streaming batch collection (`LazyFrame.collect_batches`),
so no stage ever materializes a full file in memory - only one batch (bounded by `batch_size`)
is held at a time.
"""
from __future__ import annotations

import functools
import random
from collections.abc import Iterable, Iterator, Sequence

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from uncorrupt_smiles.utils.tokenizer import smi_tokenizer
from uncorrupt_smiles.vocab import Vocab


def iter_csv_column(path: str, column: str, batch_size: int = 10_000, separator: str = ",") -> Iterator[str]:
    """Yields one string at a time from a single CSV column. Never holds more than
    `batch_size` rows in memory at once.

    The column is forced to Utf8: a SMILES column that happens to contain only
    digit-looking values (rare, but possible for a small fragment pool) would otherwise be
    silently inferred as numeric, corrupting everything downstream."""
    lf = pl.scan_csv(path, separator=separator, schema_overrides={column: pl.Utf8}).select(column)
    for batch in lf.collect_batches(chunk_size=batch_size):
        yield from batch.get_column(column).to_list()


def iter_csv_columns(
    path: str, columns: Sequence[str], batch_size: int = 10_000, separator: str = ","
) -> Iterator[tuple]:
    """Yields one row tuple at a time across multiple CSV columns (forced to Utf8, see
    iter_csv_column)."""
    lf = pl.scan_csv(
        path, separator=separator, schema_overrides={c: pl.Utf8 for c in columns}
    ).select(list(columns))
    for batch in lf.collect_batches(chunk_size=batch_size):
        yield from batch.iter_rows()


class SmilesPairIterableDataset(IterableDataset):
    """Streams (src, trg) SMILES pairs from a CSV, tokenizing/encoding on the fly.

    Shuffling in a streaming setting can't be global, so `shuffle_buffer` implements the
    standard bounded reservoir-window shuffle: fill a buffer of that size, then each new
    item swaps in for a random buffer slot while the evicted item is yielded. Set to 0 to
    disable and get file order.
    """

    def __init__(
        self,
        csv_path: str,
        src_col: str,
        trg_col: str,
        src_vocab: Vocab,
        trg_vocab: Vocab,
        batch_read_size: int = 10_000,
        shuffle_buffer: int = 0,
        seed: int = 42,
        separator: str = ",",
    ):
        self.csv_path = csv_path
        self.src_col = src_col
        self.trg_col = trg_col
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.batch_read_size = batch_read_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.separator = separator

    def _encode(self, src_text: str, trg_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.src_vocab.encode(smi_tokenizer(src_text))
        trg_ids = self.trg_vocab.encode(smi_tokenizer(trg_text, reverse=True))
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(trg_ids, dtype=torch.long),
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(self.seed + worker_id)

        rows = iter_csv_columns(
            self.csv_path, [self.src_col, self.trg_col], self.batch_read_size, self.separator
        )
        # Simple modulo sharding: every worker streams the whole file but only keeps its
        # share. Costs num_workers x redundant reads (acceptable since num_workers is
        # expected to stay small), in exchange for not needing a row-offset index.
        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, (src_text, trg_text) in enumerate(rows):
            if i % num_workers != worker_id:
                continue
            item = self._encode(src_text, trg_text)
            if self.shuffle_buffer > 0:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()
            else:
                yield item
        rng.shuffle(buffer)
        yield from buffer


def collate_pairs(
    batch: list[tuple[torch.Tensor, torch.Tensor]], src_pad_idx: int, trg_pad_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    srcs, trgs = zip(*batch)
    src_padded = pad_sequence(list(srcs), batch_first=True, padding_value=src_pad_idx)
    trg_padded = pad_sequence(list(trgs), batch_first=True, padding_value=trg_pad_idx)
    return src_padded, trg_padded


def make_loader(
    csv_path: str,
    src_col: str,
    trg_col: str,
    src_vocab: Vocab,
    trg_vocab: Vocab,
    batch_size: int,
    shuffle_buffer: int = 0,
    num_workers: int = 0,
    separator: str = ",",
    seed: int = 42,
) -> DataLoader:
    dataset = SmilesPairIterableDataset(
        csv_path, src_col, trg_col, src_vocab, trg_vocab,
        shuffle_buffer=shuffle_buffer, separator=separator, seed=seed,
    )
    collate = functools.partial(collate_pairs, src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
