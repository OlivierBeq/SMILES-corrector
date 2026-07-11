"""RDKit-based validity/complexity metrics for evaluating corrector model output.

Decoding is centralized in `decode_batch` (batch-first token-id tensors, [batch, seq_len] -
Seq2Seq's native layout) so every metric below operates on plain Python SMILES-string lists,
rather than each metric independently re-implementing tensor-to-SMILES decoding as the
original torchtext-Field-based version did.
"""
from __future__ import annotations

import statistics
from collections.abc import Iterable

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, GraphDescriptors, Lipinski

from uncorrupt_smiles.utils.chem import is_valid_smiles
from uncorrupt_smiles.vocab import Vocab

rdBase.DisableLog("rdApp.error")


def decode_batch(array: torch.Tensor, vocab: Vocab, reverse: bool) -> list[str]:
    """Decodes a batch of token-id sequences ([batch, seq_len]) into SMILES strings,
    truncating at <eos> and reversing back if the sequence was tokenized in reverse
    (matches the reverse=True convention used for target sequences during training)."""
    smiles = []
    for row in array:
        tokens = vocab.decode(row.tolist(), stop_at_eos=True)
        if reverse:
            tokens = tokens[::-1]
        smiles.append("".join(tokens))
    return smiles


def validity(smiles_list: Iterable[str]) -> list[bool]:
    """Whether each SMILES is RDKit-parseable (and non-empty)."""
    return [is_valid_smiles(s) for s in smiles_list]


def count_unchanged(sources: list[str], outputs: list[str], valids: list[bool]) -> int:
    """Count of invalid outputs that are identical to their source (model failed to change
    an already-broken input at all)."""
    return sum(1 for src, out, valid in zip(sources, outputs, valids) if not valid and out == src)


def count_reconstructed(targets: list[str], outputs: list[str]) -> int:
    """Count of outputs that are the same molecule as their target, compared via canonical
    SMILES equality. Simpler and stricter than a bidirectional substructure match (which can
    pass for non-identical molecules sharing a symmetric substructure)."""
    matches = 0
    for target, output in zip(targets, outputs):
        m = Chem.MolFromSmiles(target)
        p = Chem.MolFromSmiles(output)
        if m is None or p is None:
            continue
        if Chem.MolToSmiles(m, canonical=True) == Chem.MolToSmiles(p, canonical=True):
            matches += 1
    return matches


def complexity_whitlock(mol: Chem.Mol, include_all_descs: bool = False):
    """Complexity as defined in DOI:10.1021/jo9814546.
    S: complexity = 4*#rings + 2*#unsat + #hetatm + 2*#chiral
    Other descriptors: H: size = #bonds (incl. H), G: S + H, Ratio: S / H
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
    """Complexity as defined in DOI:10.1021/ci000145p."""
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
    targets: list[str], valids: list[bool], complexity_function=GraphDescriptors.BertzCT
) -> float:
    """Mean complexity of target molecules whose corresponding prediction was invalid -
    measures whether the model struggles more on inherently harder molecules."""
    complexities = []
    for target, valid in zip(targets, valids):
        if valid:
            continue
        m = Chem.MolFromSmiles(target)
        if m is not None:
            complexities.append(complexity_function(m))
    return statistics.mean(complexities) if complexities else 0.0


def epoch_time(start_time: float, end_time: float) -> tuple[int, int]:
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs
