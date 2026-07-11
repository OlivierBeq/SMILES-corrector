"""Transformer encoder-decoder for translating invalid SMILES into valid SMILES.

Seq2Seq owns its whole lifecycle as methods: forward (teacher-forced training step),
generate (autoregressive inference), fit/evaluate (training loop), fix_smiles/fix_smiles_csv
(the "anyone can fix a SMILES" API), and save_checkpoint/load_checkpoint (one bundled
artifact - state_dict + hyperparams + vocab, no separate config file needed).
"""
from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from uncorrupt_smiles.data import iter_csv_column
from uncorrupt_smiles.utils.metric import (
    calc_complexity,
    count_reconstructed,
    count_unchanged,
    decode_batch,
    epoch_time,
    validity,
)
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer
from uncorrupt_smiles.vocab import Vocab


def init_weights(m: nn.Module) -> None:
    """Applies Xavier-uniform init to any weighted submodule, in place.

    Intended for use with :meth:`nn.Module.apply`, e.g. ``model.apply(init_weights)``.

    :param m: Submodule being visited by :meth:`nn.Module.apply`; only
        touched if it has a multi-dimensional `weight` attribute.
    :return: None
    """
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters in a model.

    :param model: Model whose parameters are summed.
    :return: Total number of elements across all parameters with
        ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder(nn.Module):
    """Transformer encoder stack: token + positional embeddings followed by
    ``n_layers`` stacked :class:`EncoderLayer` blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        max_length: int,
        device: str,
    ) -> None:
        """
        :param input_dim: Source vocabulary size, i.e. the number of rows in
            the token embedding table.
        :param hid_dim: Model/hidden dimension carried through every sublayer.
        :param n_layers: Number of stacked :class:`EncoderLayer` blocks.
        :param n_heads: Number of attention heads per layer.
        :param pf_dim: Hidden dimension of each layer's position-wise
            feedforward sublayer.
        :param dropout: Dropout probability applied throughout the stack.
        :param max_length: Maximum sequence length supported by the
            positional embedding table.
        :param device: Device the module's embeddings and buffers are
            allocated on.
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src: Source token ids, shape ``[batch size, src len]``.
        :param src_mask: Padding mask excluding ``<pad>`` positions from
            attention, shape ``[batch size, 1, 1, src len]``.
        :return: Encoded source representations, shape
            ``[batch size, src len, hid dim]``.
        """
        batch_size, src_len = src.shape[0], src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    """One encoder block: self-attention followed by a position-wise
    feedforward sublayer, each with a residual connection and layer norm.
    """

    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str) -> None:
        """
        :param hid_dim: Model/hidden dimension carried through the block.
        :param n_heads: Number of attention heads used by the self-attention
            sublayer.
        :param pf_dim: Hidden dimension of the position-wise feedforward
            sublayer.
        :param dropout: Dropout probability applied after each sublayer.
        :param device: Device the block's buffers are allocated on.
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src: Input representations, shape ``[batch size, src len, hid dim]``.
        :param src_mask: Padding mask excluding ``<pad>`` positions from
            attention, shape ``[batch size, 1, 1, src len]``.
        :return: Updated representations, same shape as `src`.
        """
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class MultiHeadAttentionLayer(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str) -> None:
        """
        :param hid_dim: Model/hidden dimension of the query/key/value
            projections; must be divisible by `n_heads`.
        :param n_heads: Number of attention heads to split `hid_dim` into.
        :param dropout: Dropout probability applied to attention weights.
        :param device: Device the module's buffers are allocated on.
        :raises AssertionError: If `hid_dim` is not divisible by `n_heads`.
        """
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: Query vectors, shape ``[batch size, query len, hid dim]``.
        :param key: Key vectors, shape ``[batch size, key len, hid dim]``.
        :param value: Value vectors, shape ``[batch size, key len, hid dim]``.
        :param mask: Optional attention mask broadcastable to
            ``[batch size, n heads, query len, key len]``; masked positions
            (where the mask is 0) are excluded from attention.
        :return: Tuple of (attended output, shape
            ``[batch size, query len, hid dim]``; attention weights, shape
            ``[batch size, n heads, query len, key len]``).
        """
        batch_size = query.shape[0]
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """Two-layer feedforward sublayer applied independently at each position."""

    def __init__(self, hid_dim: int, pf_dim: int, dropout: float) -> None:
        """
        :param hid_dim: Input/output dimension of the sublayer.
        :param pf_dim: Hidden dimension of the intermediate linear layer.
        :param dropout: Dropout probability applied after the ReLU.
        """
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input, shape ``[batch size, seq len, hid dim]``.
        :return: Output, same shape as `x`.
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    """Transformer decoder stack: token + positional embeddings, ``n_layers``
    stacked :class:`DecoderLayer` blocks, and an output projection to vocabulary
    logits.
    """

    def __init__(
        self,
        output_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        max_length: int,
        device: str,
    ) -> None:
        """
        :param output_dim: Target vocabulary size, i.e. the number of logits
            produced per position.
        :param hid_dim: Model/hidden dimension carried through every sublayer.
        :param n_layers: Number of stacked :class:`DecoderLayer` blocks.
        :param n_heads: Number of attention heads per layer.
        :param pf_dim: Hidden dimension of each layer's position-wise
            feedforward sublayer.
        :param dropout: Dropout probability applied throughout the stack.
        :param max_length: Maximum sequence length supported by the
            positional embedding table.
        :param device: Device the module's embeddings and buffers are
            allocated on.
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(
        self,
        trg: torch.Tensor,
        enc_src: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        :param trg: Target token ids, shape ``[batch size, trg len]``.
        :param enc_src: Encoder output to attend over, shape
            ``[batch size, src len, hid dim]``.
        :param trg_mask: Combined padding + causal mask for `trg`, shape
            ``[batch size, 1, trg len, trg len]``.
        :param src_mask: Padding mask for `enc_src`, shape
            ``[batch size, 1, 1, src len]``.
        :return: Tuple of (vocabulary logits, shape
            ``[batch size, trg len, output dim]``; last layer's
            encoder-decoder attention weights).
        """
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    """One decoder block: masked self-attention, encoder-decoder attention, then
    a position-wise feedforward sublayer, each with a residual connection and
    layer norm.
    """

    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str) -> None:
        """
        :param hid_dim: Model/hidden dimension carried through the block.
        :param n_heads: Number of attention heads used by each attention
            sublayer.
        :param pf_dim: Hidden dimension of the position-wise feedforward
            sublayer.
        :param dropout: Dropout probability applied after each sublayer.
        :param device: Device the block's buffers are allocated on.
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: torch.Tensor,
        enc_src: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param trg: Target representations, shape ``[batch size, trg len, hid dim]``.
        :param enc_src: Encoder output to attend over, shape
            ``[batch size, src len, hid dim]``.
        :param trg_mask: Combined padding + causal mask for `trg`, shape
            ``[batch size, 1, trg len, trg len]``.
        :param src_mask: Padding mask for `enc_src`, shape
            ``[batch size, 1, 1, src len]``.
        :return: Tuple of (updated target representations, same shape as
            `trg`; encoder-decoder attention weights, shape
            ``[batch size, n heads, trg len, src len]``).
        """
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention


class Seq2Seq(nn.Module):
    """Encoder-decoder transformer that translates invalid SMILES into valid ones.

    Owns its whole lifecycle as methods: :meth:`forward` (teacher-forced
    training step), :meth:`generate` (autoregressive inference),
    :meth:`fit`/:meth:`evaluate` (training loop), :meth:`fix_smiles`/
    :meth:`fix_smiles_csv` (the "anyone can fix a SMILES" API), and
    :meth:`save_checkpoint`/:meth:`load_checkpoint` (one bundled artifact -
    state_dict + hyperparams + vocab, no separate config file needed).
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int,
        trg_pad_idx: int,
        device: str,
        hyperparams: dict,
        src_vocab: Vocab | None = None,
        trg_vocab: Vocab | None = None,
    ) -> None:
        """
        :param encoder: Encoder submodule that turns source ids into
            contextual representations.
        :param decoder: Decoder submodule that turns those representations
            into target vocabulary logits.
        :param src_pad_idx: Padding token id in the source vocabulary, used
            to build the source attention mask.
        :param trg_pad_idx: Padding token id in the target vocabulary, used
            to build the target attention mask.
        :param device: Device the model runs on.
        :param hyperparams: Architecture hyperparameters, as produced by
            :meth:`build`; persisted verbatim by :meth:`save_checkpoint` so
            :meth:`load_checkpoint` can reconstruct an identical architecture.
        :param src_vocab: Vocabulary token ids are encoded against on the
            source (input) side. Prefer setting this via :meth:`build`,
            :meth:`fit`, or :meth:`load_checkpoint` rather than directly.
        :param trg_vocab: Vocabulary token ids are decoded against on the
            target (output) side.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.hyperparams = hyperparams
        # The vocab a checkpoint was trained with is not optional configuration - encoding/
        # decoding with a mismatched vocab silently produces garbage. Carrying it as part of
        # the model (rather than a value the caller must separately track and pass to every
        # call) makes that mismatch impossible.
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        max_length: int,
        device: str,
        src_pad_idx: int,
        trg_pad_idx: int,
        hid_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        pf_dim: int = 256,
        dropout: float = 0.1,
        src_vocab: Vocab | None = None,
        trg_vocab: Vocab | None = None,
    ) -> "Seq2Seq":
        """Constructs an encoder, decoder, and wrapping :class:`Seq2Seq`, then
        applies :func:`init_weights` and moves the result to `device`.

        Small-but-functional defaults sized for an 8GB GPU; a full-scale run
        just passes larger `hid_dim`/`n_layers`/`n_heads`/`pf_dim`.

        :param input_dim: Source vocabulary size.
        :param output_dim: Target vocabulary size.
        :param max_length: Maximum sequence length supported by the
            positional embeddings.
        :param device: Device to build the model on.
        :param src_pad_idx: Padding token id in the source vocabulary.
        :param trg_pad_idx: Padding token id in the target vocabulary.
        :param hid_dim: Model/hidden dimension.
        :param n_layers: Number of encoder/decoder layers.
        :param n_heads: Number of attention heads per layer.
        :param pf_dim: Hidden dimension of each position-wise feedforward
            sublayer.
        :param dropout: Dropout probability.
        :param src_vocab: Source vocabulary to attach to the model, if
            available.
        :param trg_vocab: Target vocabulary to attach to the model, if
            available.
        :return: A newly initialized model on `device`.
        """
        encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length, device)
        decoder = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length, device)
        hyperparams = dict(
            input_dim=input_dim, output_dim=output_dim, max_length=max_length,
            src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, hid_dim=hid_dim,
            n_layers=n_layers, n_heads=n_heads, pf_dim=pf_dim, dropout=dropout,
        )
        model = cls(encoder, decoder, src_pad_idx, trg_pad_idx, device, hyperparams, src_vocab, trg_vocab)
        model.apply(init_weights)
        return model.to(device)

    def _require_vocab(self) -> tuple[Vocab, Vocab]:
        """Guard used by methods that need `src_vocab`/`trg_vocab` to already be set.

        :return: The model's source and target vocabularies.
        :raises ValueError: If either `src_vocab` or `trg_vocab` is unset.
        """
        if self.src_vocab is None or self.trg_vocab is None:
            raise ValueError(
                "this model has no src_vocab/trg_vocab attached - set them via fit(...), "
                "load_checkpoint(...), or by assigning model.src_vocab/model.trg_vocab directly"
            )
        return self.src_vocab, self.trg_vocab

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Builds the source padding mask used to exclude pad tokens from attention.

        :param src: Source token ids, shape ``[batch size, src len]``.
        :return: Boolean mask, shape ``[batch size, 1, 1, src len]``, ``True``
            at non-pad positions.
        """
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Builds the target mask combining padding exclusion with causal
        (no-peek-ahead) masking.

        :param trg: Target token ids, shape ``[batch size, trg len]``.
        :return: Boolean mask, shape ``[batch size, 1, trg len, trg len]``,
            ``True`` where a position may attend to another.
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Teacher-forced training step.

        :param src: Source token ids, shape ``[batch size, src len]``.
        :param trg: Decoder-input target token ids, shape
            ``[batch size, trg len]`` (i.e. the caller passes ``trg[:, :-1]``,
            not the full target sequence).
        :return: Tuple of (vocabulary logits, shape
            ``[batch size, trg len, output dim]``; last decoder layer's
            encoder-decoder attention weights).
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int, sos_idx: int, eos_idx: int) -> torch.Tensor:
        """Greedy autoregressive decode.

        :param src: Source token ids to condition generation on, shape
            ``[batch size, src len]``.
        :param max_len: Maximum number of tokens to generate beyond the
            leading ``<sos>``.
        :param sos_idx: Id of the start-of-sequence token to seed decoding with.
        :param eos_idx: Id of the end-of-sequence token; generation for a
            given sequence stops once this token is emitted.
        :return: Generated token ids, shape ``[batch size, out len]``,
            including the leading `sos_idx`. Decoding stops early once every
            sequence in the batch has emitted `eos_idx`.
        """
        self.eval()
        batch_size = src.shape[0]
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg)
            output, _ = self.decoder(trg, enc_src, trg_mask, src_mask)
            next_token = output[:, -1, :].argmax(-1, keepdim=True)
            trg = torch.cat([trg, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break
        return trg

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        src_vocab: Vocab,
        trg_vocab: Vocab,
        epochs: int,
        lr: float = 5e-4,
        clip: float = 0.1,
        checkpoint_path: str | None = None,
        patience: int = 10,
        max_len: int | None = None,
    ) -> None:
        """Owns the whole training loop: optimizer, loss, gradient clipping,
        per-epoch validation metrics, and best-checkpoint tracking.

        Checkpoints whenever the reconstruction error improves; early-stops
        after `patience` epochs without improvement if `valid_loader` is
        given. Attaches `src_vocab`/`trg_vocab` to the model itself, since a
        checkpoint and the vocab it was trained with cannot be separated.

        :param train_loader: Batches of ``(src, trg)`` tensors to train on.
        :param valid_loader: Batches of ``(src, trg)`` tensors to validate
            on after each epoch. If ``None``, training loss is used for
            best-checkpoint tracking instead and early stopping is disabled.
        :param src_vocab: Source vocabulary, attached to the model for the
            duration of training and beyond.
        :param trg_vocab: Target vocabulary, attached to the model for the
            duration of training and beyond.
        :param epochs: Maximum number of epochs to train for.
        :param lr: Adam learning rate.
        :param clip: Max gradient norm for gradient clipping.
        :param checkpoint_path: If given, path to write the best checkpoint
            to via :meth:`save_checkpoint`.
        :param patience: Number of epochs without improvement (only
            evaluated when `valid_loader` is given) before stopping early.
        :param max_len: Maximum generation length used during validation;
            defaults to ``hyperparams["max_length"]`` if ``None``.
        :return: None
        """
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        print(f"training model with {count_parameters(self):,} trainable parameters "
              f"on {self.device}")

        best_error = float("inf")
        epochs_since_improve = 0
        for epoch in range(epochs):
            self.train()
            start_time = time.time()
            train_loss = 0.0
            n_train_batches = 0
            for src, trg in train_loader:
                src, trg = src.to(self.device), trg.to(self.device)
                optimizer.zero_grad()
                output, _ = self(src, trg[:, :-1])
                output_dim = output.shape[-1]
                loss = criterion(
                    output.contiguous().view(-1, output_dim),
                    trg[:, 1:].contiguous().view(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                train_loss += loss.item()
                n_train_batches += 1
            train_loss /= max(n_train_batches, 1)
            mins, secs = epoch_time(start_time, time.time())
            info = f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4g} time={mins}m{secs}s"

            if valid_loader is not None:
                metrics = self.evaluate(valid_loader, criterion, max_len)
                error = 1 - metrics["reconstruction_rate"]
                info += (
                    f" valid_loss={metrics['loss']:.4g}"
                    f" validity={metrics['validity_rate']:.4g}"
                    f" reconstruction={metrics['reconstruction_rate']:.4g}"
                    f" unchanged={metrics['unchanged_rate']:.4g}"
                    f" complexity={metrics['complexity']:.4g}"
                )
                if error < best_error:
                    best_error = error
                    epochs_since_improve = 0
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                else:
                    epochs_since_improve += 1
            elif train_loss < best_error:
                best_error = train_loss
                epochs_since_improve = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)

            print(info)
            if valid_loader is not None and epochs_since_improve >= patience:
                print(f"no improvement in {patience} epochs, stopping early")
                break

    def evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module | None = None,
        max_len: int | None = None,
    ) -> dict[str, float]:
        """Runs teacher-forced loss plus free-running :meth:`generate` over
        `loader`, returning validity/reconstruction/unchanged rates and
        target-complexity-of-failures. Uses the model's own
        `src_vocab`/`trg_vocab` (set by :meth:`fit` or :meth:`load_checkpoint`).

        :param loader: Batches of ``(src, trg)`` tensors to evaluate on.
        :param criterion: Loss function; defaults to
            ``nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)`` if ``None``.
        :param max_len: Maximum generation length; defaults to
            ``hyperparams["max_length"]`` if ``None``.
        :return: Dict with keys ``loss``, ``validity_rate``,
            ``reconstruction_rate``, ``unchanged_rate``, and ``complexity``.
        :raises ValueError: If the model has no `src_vocab`/`trg_vocab` attached.
        """
        src_vocab, trg_vocab = self._require_vocab()
        self.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        if max_len is None:
            max_len = self.hyperparams["max_length"]

        total_loss = 0.0
        n_batches = 0
        n_total = 0
        n_valid = 0
        n_reconstructed = 0
        n_unchanged = 0
        n_invalid = 0
        batch_complexities = []
        with torch.no_grad():
            for src, trg in loader:
                src, trg = src.to(self.device), trg.to(self.device)
                output, _ = self(src, trg[:, :-1])
                output_dim = output.shape[-1]
                loss = criterion(
                    output.contiguous().view(-1, output_dim),
                    trg[:, 1:].contiguous().view(-1),
                )
                total_loss += loss.item()
                n_batches += 1

                gen_ids = self.generate(src, max_len, trg_vocab.sos_idx, trg_vocab.eos_idx)
                outputs = decode_batch(gen_ids, trg_vocab, reverse=True)
                targets = decode_batch(trg, trg_vocab, reverse=True)
                sources = decode_batch(src, src_vocab, reverse=False)

                valids = validity(outputs)
                n_valid += sum(valids)
                n_total += len(outputs)
                n_invalid += sum(1 for v in valids if not v)
                n_reconstructed += count_reconstructed(targets, outputs)
                n_unchanged += count_unchanged(sources, outputs, valids)
                batch_complexities.append(calc_complexity(targets, valids))

        return {
            "loss": total_loss / max(n_batches, 1),
            "validity_rate": n_valid / max(n_total, 1),
            "reconstruction_rate": n_reconstructed / max(n_total, 1),
            "unchanged_rate": n_unchanged / max(n_invalid, 1),
            "complexity": statistics.mean(batch_complexities) if batch_complexities else 0.0,
        }

    def fix_smiles(
        self,
        smiles: str | Iterable[str],
        max_len: int | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        """The "anyone can fix a SMILES" entrypoint. Uses the model's own
        `src_vocab`/`trg_vocab` (set by :meth:`fit` or :meth:`load_checkpoint`).

        :param smiles: A single (possibly invalid) SMILES string, or any
            iterable of them (list, generator), to correct.
        :param max_len: Maximum generation length per SMILES; defaults to
            ``hyperparams["max_length"]`` if ``None``.
        :param batch_size: Number of SMILES encoded/generated per batch.
        :return: Corrected SMILES, in the same order as the input.
        :raises ValueError: If the model has no `src_vocab`/`trg_vocab` attached.
        """
        src_vocab, trg_vocab = self._require_vocab()
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        if max_len is None:
            max_len = self.hyperparams["max_length"]
        self.eval()
        results: list[str] = []
        for start in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[start:start + batch_size]
            encoded = [src_vocab.as_tensor(smi_tokenizer(s), device=self.device) for s in chunk]
            src = pad_sequence(encoded, batch_first=True, padding_value=src_vocab.pad_idx)
            gen_ids = self.generate(src, max_len, trg_vocab.sos_idx, trg_vocab.eos_idx)
            results.extend(decode_batch(gen_ids, trg_vocab, reverse=True))
        return results

    def fix_smiles_csv(
        self,
        input_csv: str,
        smiles_col: str,
        output_csv: str,
        batch_size: int = 64,
        separator: str = ",",
    ) -> None:
        """Streams `input_csv` in batches, calls :meth:`fix_smiles` per batch,
        and writes results incrementally. Works on any single-column SMILES
        file (e.g. output from an external generative model) with zero
        full-file read.

        :param input_csv: Path to the CSV file containing the (possibly
            invalid) SMILES to correct.
        :param smiles_col: Name of the column in `input_csv` to read SMILES from.
        :param output_csv: Destination CSV path to write results to; written
            with columns `smiles_col` (original SMILES) and ``"FIXED"``
            (corrected SMILES).
        :param batch_size: Number of SMILES fixed per batch.
        :param separator: Field separator used by `input_csv`.
        :return: None
        """
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([smiles_col, "FIXED"])
            chunk: list[str] = []
            for smi in iter_csv_column(input_csv, smiles_col, separator=separator):
                chunk.append(smi)
                if len(chunk) >= batch_size:
                    for orig, fixed in zip(chunk, self.fix_smiles(chunk, batch_size=batch_size)):
                        writer.writerow([orig, fixed])
                    chunk = []
            if chunk:
                for orig, fixed in zip(chunk, self.fix_smiles(chunk, batch_size=batch_size)):
                    writer.writerow([orig, fixed])

    def save_checkpoint(self, path: str) -> None:
        """Bundles state_dict + architecture hyperparams + vocab into one
        artifact - a learned/data artifact (like any PyTorch checkpoint), not
        a hand-authored config file. Uses the model's own
        `src_vocab`/`trg_vocab` (set by :meth:`fit` or :meth:`load_checkpoint`).

        :param path: Destination file path to write the checkpoint to,
            readable back via :meth:`load_checkpoint`.
        :return: None
        :raises ValueError: If the model has no `src_vocab`/`trg_vocab` attached.
        """
        src_vocab, trg_vocab = self._require_vocab()
        torch.save({
            "hyperparams": self.hyperparams,
            "state_dict": self.state_dict(),
            "src_itos": src_vocab.itos,
            "trg_itos": trg_vocab.itos,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str) -> "Seq2Seq":
        """Reconstructs a model together with the `src_vocab`/`trg_vocab` it
        was trained with - the two cannot be used correctly apart from one
        another, so they are attached to the returned model rather than
        handed back separately.

        :param path: Path to a checkpoint file previously written by
            :meth:`save_checkpoint`.
        :param device: Device to load the reconstructed model onto.
        :return: The reconstructed model, in evaluation-ready state.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        src_vocab = Vocab(checkpoint["src_itos"])
        trg_vocab = Vocab(checkpoint["trg_itos"])
        model = cls.build(device=device, src_vocab=src_vocab, trg_vocab=trg_vocab, **checkpoint["hyperparams"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
