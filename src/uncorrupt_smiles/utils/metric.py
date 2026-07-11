"""RDKit-based validity/complexity metrics for evaluating corrector model output.

Decoding is centralized in `decode_batch` (batch-first token-id tensors, [batch, seq_len] -
Seq2Seq's native layout) so every metric below operates on plain Python SMILES-string lists,
rather than each metric independently re-implementing tensor-to-SMILES decoding as the
original torchtext-Field-based version did.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, GraphDescriptors, Lipinski

from uncorrupt_smiles.utils.chem import is_valid_smiles
from uncorrupt_smiles.vocab import Vocab

rdBase.DisableLog("rdApp.error")


def decode_batch(array: torch.Tensor, vocab: Vocab, reverse: bool) -> list[str]:
    """Decodes a batch of token-id sequences into SMILES strings.

    :param array: Batch-first token-id tensor, shape ``[batch, seq_len]``
        (Seq2Seq's native layout).
    :param vocab: Vocabulary used to decode ids back to tokens; decoding
        truncates at ``<eos>``.
    :param reverse: If ``True``, reverse each decoded token sequence before
        joining (matches the ``reverse=True`` convention used for target
        sequences during training).
    :return: Decoded SMILES strings, one per row of `array`.
    """
    smiles = []
    for row in array:
        tokens = vocab.decode(row.tolist(), stop_at_eos=True)
        if reverse:
            tokens = tokens[::-1]
        smiles.append("".join(tokens))
    return smiles


def validity(smiles_list: Iterable[str]) -> list[bool]:
    """Checks each SMILES for RDKit-parseability.

    :param smiles_list: SMILES strings to validate.
    :return: ``True`` per input that is RDKit-parseable and non-empty, via
        :func:`~uncorrupt_smiles.utils.chem.is_valid_smiles`.
    """
    return [is_valid_smiles(s) for s in smiles_list]


def count_unchanged(sources: list[str], outputs: list[str], valids: list[bool]) -> int:
    """Counts invalid outputs identical to their source.

    :param sources: Input SMILES fed to the model.
    :param outputs: Model-produced SMILES, aligned index-wise with `sources`.
    :param valids: Validity flags for `outputs`, aligned index-wise (e.g. from
        :func:`validity`).
    :return: Number of examples where the output is invalid and identical to
        the source, i.e. the model failed to change an already-broken input
        at all.
    """
    return sum(1 for src, out, valid in zip(sources, outputs, valids) if not valid and out == src)


def count_reconstructed(targets: list[str], outputs: list[str]) -> int:
    """Counts outputs that are the same molecule as their target.

    Compares via canonical SMILES equality - simpler and stricter than a
    bidirectional substructure match, which can pass for non-identical
    molecules sharing a symmetric substructure.

    :param targets: Expected/ground-truth SMILES.
    :param outputs: Model-produced SMILES, aligned index-wise with `targets`.
    :return: Number of index-aligned pairs whose canonical SMILES match.
        Pairs where either side fails to parse are skipped.
    """
    matches = 0
    for target, output in zip(targets, outputs):
        m = Chem.MolFromSmiles(target)
        p = Chem.MolFromSmiles(output)
        if m is None or p is None:
            continue
        if Chem.MolToSmiles(m, canonical=True) == Chem.MolToSmiles(p, canonical=True):
            matches += 1
    return matches


def complexity_whitlock(mol: Chem.Mol, include_all_descs: bool = False) -> int | dict[str, float]:
    """Computes molecular complexity as defined in DOI:10.1021/jo9814546.

    S: complexity = 4*#rings + 2*#unsat + #hetatm + 2*#chiral.
    Other descriptors: H: size = #bonds (incl. H), G: S + H, Ratio: S / H.

    :param mol: Molecule to score. Not mutated - an internal copy is used.
    :param include_all_descs: If ``True``, also compute H, G and Ratio in
        addition to S.
    :return: The S-score alone if `include_all_descs` is ``False``, otherwise
        a dict with keys ``"WhitlockS"``, ``"WhitlockH"``, ``"WhitlockG"``,
        ``"WhitlockRatio"``.
    """
    mol_ = Chem.Mol(mol)
    nrings = Lipinski.RingCount(mol_) - Lipinski.NumAromaticRings(mol_)
    Chem.rdmolops.SetAromaticity(mol_)
    unsat = sum(1 for bond in mol_.GetBonds() if bond.GetBondTypeAsDouble() == 2)
    hetatm = len(mol_.GetSubstructMatches(Chem.MolFromSmarts("[!#6]")))
    AllChem.EmbedMolecule(mol_)
    Chem.AssignAtomChiralTagsFromStructure(mol_)
    chiral = len(Chem.FindMolChiralCenters(mol_))
    s_score = 4 * nrings + 2 * unsat + hetatm + 2 * chiral
    if not include_all_descs:
        return s_score
    Chem.rdmolops.Kekulize(mol_)
    mol_ = Chem.AddHs(mol_)
    h_score = sum(bond.GetBondTypeAsDouble() for bond in mol_.GetBonds())
    return {
        "WhitlockS": s_score,
        "WhitlockH": h_score,
        "WhitlockG": s_score + h_score,
        "WhitlockRatio": s_score / h_score,
    }


def complexity_baronechanon(mol: Chem.Mol) -> float:
    """Computes molecular complexity as defined in DOI:10.1021/ci000145p.

    :param mol: Molecule to score. Not mutated - an internal copy is used.
    :return: Complexity score.
    """
    mol_ = Chem.Mol(mol)
    Chem.Kekulize(mol_)
    Chem.RemoveStereochemistry(mol_)
    mol_ = Chem.RemoveHs(mol_, updateExplicitCount=True)
    degree, counts = 0, 0
    for atom in mol_.GetAtoms():
        degree += 3 * 2 ** (atom.GetExplicitValence() - atom.GetNumExplicitHs() - 1)
        counts += 3 if atom.GetSymbol() == "C" else 6
    ringterm = sum(6 * len(ring) for ring in mol_.GetRingInfo().AtomRings())
    return degree + counts + ringterm


def calc_complexity(
    targets: list[str],
    valids: list[bool],
    complexity_function: Callable[[Chem.Mol], float] = GraphDescriptors.BertzCT,
) -> float:
    """Computes mean complexity of target molecules whose corresponding
    prediction was invalid - measures whether the model struggles more on
    inherently harder molecules.

    :param targets: Expected/ground-truth SMILES.
    :param valids: Validity flags for the corresponding predictions, aligned
        index-wise with `targets` (e.g. from :func:`validity`).
    :param complexity_function: Callable computing a complexity score for an
        RDKit molecule; defaults to RDKit's Bertz complexity index
        (``GraphDescriptors.BertzCT``).
    :return: Mean complexity over targets whose prediction was invalid and
        which parse successfully, or ``0.0`` if there are none.
    """
    complexities = []
    for target, valid in zip(targets, valids):
        if valid:
            continue
        m = Chem.MolFromSmiles(target)
        if m is not None:
            complexities.append(complexity_function(m))
    return statistics.mean(complexities) if complexities else 0.0


def epoch_time(start_time: float, end_time: float) -> tuple[int, int]:
    """Converts an elapsed duration into whole minutes and seconds.

    :param start_time: Start timestamp (e.g. from :func:`time.time`).
    :param end_time: End timestamp, same clock as `start_time`.
    :return: ``(minutes, seconds)``, both truncated to whole numbers.
    """
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs
