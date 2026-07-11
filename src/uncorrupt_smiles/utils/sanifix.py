"""Sanifix4 by James Davidson - fixes aromatic nitrogen perception RDKit sometimes gets wrong
(e.g. O=c1ccncc1). Used by invalid_smiles.arom_error."""
from rdkit import Chem


def _frag_indices_to_mol(o_mol, indices):
    em = Chem.rdchem.EditableMol(Chem.rdchem.Mol())

    new_indices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(o_mol.GetAtomWithIdx(idx))
        new_indices[idx] = i

    for idx in indices:
        at = o_mol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once
            if oidx < idx:
                continue
            em.AddBond(new_indices[idx], new_indices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.rdmolops.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = new_indices
    return res


def _recursively_modify_ns(mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    matches = list(matches)
    while len(matches) and res is None:
        t_indices = indices[:]
        next_idx = matches.pop(0)
        t_indices.append(next_idx)
        nm = Chem.rdchem.Mol(mol)
        nm.GetAtomWithIdx(next_idx).SetNoImplicit(True)
        nm.GetAtomWithIdx(next_idx).SetNumExplicitHs(0)
        cp = Chem.rdchem.Mol(nm)
        try:
            Chem.rdmolops.SanitizeMol(cp)
            res, indices = _recursively_modify_ns(nm, matches, indices=t_indices)
        except ValueError:
            indices = t_indices
            res = cp
    return res, indices


def AdjustAromaticNs(m, nitrogen_pattern="[n&D2&H1;r5,r6]"):
    """Default nitrogen pattern matches Ns in 5- and 6-rings so molecules such as
    O=c1ccncc1 can be sanitized/aromatized correctly."""
    Chem.rdmolops.GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings
    em = Chem.rdchem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.rdmolfiles.MolFromSmarts("[r]!@[r]"))
    pls_fix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        pls_fix.add(a)
        pls_fix.add(b)
    nm = em.GetMol()
    for at_idx in pls_fix:
        at = nm.GetAtomWithIdx(at_idx)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments
    frag_lists = Chem.rdmolops.GetMolFrags(nm)
    frags = [_frag_indices_to_mol(nm, x) for x in frag_lists]

    # loop through the fragments in turn and try to aromatize them
    for frag in frags:
        matches = [
            x[0] for x in frag.GetSubstructMatches(
                Chem.rdmolfiles.MolFromSmarts(nitrogen_pattern))
        ]
        lres, indices = _recursively_modify_ns(frag, matches)
        if not lres:
            break
        rev_map = {v: k for k, v in frag._idxMap.items()}
        for idx in indices:
            oatom = m.GetAtomWithIdx(rev_map[idx])
            oatom.SetNoImplicit(True)
            oatom.SetNumExplicitHs(0)

    return m
