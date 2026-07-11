from types import SimpleNamespace

import torch

from uncorrupt_smiles.data import (
    SmilesPairIterableDataset,
    collate_pairs,
    iter_csv_column,
    iter_csv_columns,
    make_loader,
)
from uncorrupt_smiles.vocab import Vocab

PAIRS = [("CCO", "OCC"), ("c1ccccc1", "C1=CC=CC=C1"), ("CCN", "NCC")]


def _pair_csv(tmp_path, pairs=PAIRS):
    import csv
    path = tmp_path / "pairs.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ERROR", "STD_SMILES"])
        for src, trg in pairs:
            w.writerow([src, trg])
    return str(path)


def test_iter_csv_column_preserves_order(tmp_path):
    path = _pair_csv(tmp_path)
    values = list(iter_csv_column(path, "ERROR", batch_size=1))
    assert values == [p[0] for p in PAIRS]


def test_iter_csv_columns_yields_row_tuples(tmp_path):
    path = _pair_csv(tmp_path)
    rows = list(iter_csv_columns(path, ["ERROR", "STD_SMILES"], batch_size=2))
    assert rows == PAIRS


def _vocabs():
    src_vocab = Vocab.build_from_lines([p[0] for p in PAIRS], max_size=50)
    trg_vocab = Vocab.build_from_lines([p[1] for p in PAIRS], max_size=50)
    return src_vocab, trg_vocab


def test_dataset_items_have_sos_eos(tmp_path):
    path = _pair_csv(tmp_path)
    src_vocab, trg_vocab = _vocabs()
    ds = SmilesPairIterableDataset(path, "ERROR", "STD_SMILES", src_vocab, trg_vocab)
    items = list(ds)
    assert len(items) == len(PAIRS)
    for src_ids, trg_ids in items:
        assert src_ids[0].item() == src_vocab.sos_idx
        assert src_ids[-1].item() == src_vocab.eos_idx
        assert trg_ids[0].item() == trg_vocab.sos_idx
        assert trg_ids[-1].item() == trg_vocab.eos_idx


def test_worker_sharding_covers_every_row_exactly_once(tmp_path, monkeypatch):
    path = _pair_csv(tmp_path, pairs=PAIRS * 4)  # 12 rows, enough to split across 2 workers
    src_vocab, trg_vocab = _vocabs()

    def _collect_for_worker(worker_id, num_workers):
        monkeypatch.setattr(
            "uncorrupt_smiles.data.get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=num_workers),
        )
        ds = SmilesPairIterableDataset(path, "ERROR", "STD_SMILES", src_vocab, trg_vocab)
        return list(ds)

    shard0 = _collect_for_worker(0, 2)
    shard1 = _collect_for_worker(1, 2)
    assert len(shard0) + len(shard1) == len(PAIRS) * 4
    # no overlap, no gaps: every row goes to exactly one worker
    assert len(shard0) == len(PAIRS) * 4 - len(shard1)


def test_collate_pairs_pads_to_longest_in_batch():
    src_vocab, trg_vocab = _vocabs()
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2])),
        (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3])),
    ]
    src, trg = collate_pairs(batch, src_vocab.pad_idx, trg_vocab.pad_idx)
    assert src.shape == (2, 5)
    assert trg.shape == (2, 3)
    assert src[0, -1].item() == src_vocab.pad_idx
    assert trg[0, -1].item() == trg_vocab.pad_idx


def test_make_loader_yields_correct_batch_sizes(tmp_path):
    path = _pair_csv(tmp_path, pairs=PAIRS * 3)  # 9 rows
    src_vocab, trg_vocab = _vocabs()
    loader = make_loader(path, "ERROR", "STD_SMILES", src_vocab, trg_vocab, batch_size=4)
    batch_sizes = [src.shape[0] for src, _ in loader]
    assert batch_sizes == [4, 4, 1]


def test_shuffle_buffer_preserves_multiset(tmp_path):
    path = _pair_csv(tmp_path, pairs=PAIRS * 5)
    src_vocab, trg_vocab = _vocabs()
    ds_shuffled = SmilesPairIterableDataset(
        path, "ERROR", "STD_SMILES", src_vocab, trg_vocab, shuffle_buffer=5, seed=1
    )
    ds_plain = SmilesPairIterableDataset(path, "ERROR", "STD_SMILES", src_vocab, trg_vocab)
    shuffled_srcs = sorted(tuple(x[0].tolist()) for x in ds_shuffled)
    plain_srcs = sorted(tuple(x[0].tolist()) for x in ds_plain)
    assert shuffled_srcs == plain_srcs
