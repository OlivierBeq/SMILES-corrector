import torch

from uncorrupt_smiles.data import iter_csv_column, make_loader
from uncorrupt_smiles.transformer import Seq2Seq
from uncorrupt_smiles.vocab import Vocab

TINY_KWARGS = dict(hid_dim=16, n_layers=1, n_heads=2, pf_dim=32, dropout=0.1)


def _build_vocabs(train_csv):
    src_vocab = Vocab.build_from_lines(iter_csv_column(train_csv, "ERROR"), max_size=100)
    trg_vocab = Vocab.build_from_lines(iter_csv_column(train_csv, "STD_SMILES"), max_size=100)
    return src_vocab, trg_vocab


def test_build_constructs_model(errors_csv_pair):
    train_csv, _ = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    assert sum(p.numel() for p in model.parameters()) > 0
    assert model.hyperparams["hid_dim"] == 16


def test_generate_returns_ids_starting_with_sos(errors_csv_pair):
    train_csv, _ = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    src = torch.tensor([[src_vocab.sos_idx, src_vocab.stoi["C"], src_vocab.eos_idx]])
    out = model.generate(src, max_len=10, sos_idx=trg_vocab.sos_idx, eos_idx=trg_vocab.eos_idx)
    assert out[0, 0].item() == trg_vocab.sos_idx
    assert out.shape[1] <= 11  # sos + up to max_len generated tokens


def test_fit_runs_and_loss_is_finite(errors_csv_pair):
    train_csv, dev_csv = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    train_loader = make_loader(train_csv, "ERROR", "STD_SMILES", src_vocab, trg_vocab, batch_size=4)
    dev_loader = make_loader(dev_csv, "ERROR", "STD_SMILES", src_vocab, trg_vocab, batch_size=4)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    model.fit(train_loader, dev_loader, src_vocab, trg_vocab, epochs=2, checkpoint_path=None)
    metrics = model.evaluate(dev_loader)
    assert metrics["loss"] == metrics["loss"]  # not NaN
    assert 0.0 <= metrics["validity_rate"] <= 1.0


def test_checkpoint_round_trip_reproduces_output(errors_csv_pair, tmp_path):
    train_csv, _ = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    model.src_vocab, model.trg_vocab = src_vocab, trg_vocab
    ckpt_path = str(tmp_path / "ckpt.pkg")
    model.save_checkpoint(ckpt_path)

    model2 = Seq2Seq.load_checkpoint(ckpt_path, device="cpu")
    assert model2.src_vocab.itos == src_vocab.itos
    assert model2.trg_vocab.itos == trg_vocab.itos
    assert model2.hyperparams == model.hyperparams

    model.eval()
    model2.eval()
    out1 = model.fix_smiles(["CCO"], max_len=10)
    out2 = model2.fix_smiles(["CCO"], max_len=10)
    assert out1 == out2


def test_fix_smiles_accepts_string_and_list(errors_csv_pair):
    train_csv, _ = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    model.src_vocab, model.trg_vocab = src_vocab, trg_vocab
    single = model.fix_smiles("CCO", max_len=10)
    multi = model.fix_smiles(["CCO", "c1ccccc1"], max_len=10)
    assert isinstance(single, list) and len(single) == 1
    assert isinstance(multi, list) and len(multi) == 2
    assert "<pad>" not in single[0] and "<sos>" not in single[0]


def test_fix_smiles_csv_writes_one_row_per_input(errors_csv_pair, smiles_csv, tmp_path):
    train_csv, _ = errors_csv_pair
    src_vocab, trg_vocab = _build_vocabs(train_csv)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    model.src_vocab, model.trg_vocab = src_vocab, trg_vocab
    output_csv = str(tmp_path / "fixed.csv")
    model.fix_smiles_csv(smiles_csv, "SMILES", output_csv, batch_size=4)

    import csv as csv_mod
    with open(smiles_csv) as f:
        n_input = sum(1 for _ in csv_mod.DictReader(f))
    with open(output_csv) as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == n_input
