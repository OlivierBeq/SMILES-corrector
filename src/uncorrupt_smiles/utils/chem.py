from rdkit import Chem


def is_valid_smiles(smiles) -> bool:
    """True if `smiles` is a non-empty string RDKit can parse into an actual molecule.

    RDKit's Chem.MolFromSmiles("") returns a valid zero-atom Mol rather than None, so a bare
    `MolFromSmiles(s) is not None` check silently treats an empty string as "valid" - this
    helper closes that gap, since an empty SMILES represents no molecule at all.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() > 0
