"""Introduce synthetic errors into valid SMILES, for training a corrector model.

The per-SMILES error functions (`exists_error`, `par_error`, `permutation`, `ring_error`,
`syntax_error`, `valence_error`, `arom_error`, `introduce_error`) are unchanged from the
original implementation (each already operates on a single SMILES string). Everything above
them is new: streaming orchestration that works on any CSV without loading it fully into RAM.
"""
from __future__ import annotations

import csv
import random
import re
from collections.abc import Iterable, Iterator

from rdkit import Chem, RDLogger

from uncorrupt_smiles.data import iter_csv_column
from uncorrupt_smiles.utils.chem import is_valid_smiles
from uncorrupt_smiles.utils.sanifix import adjust_aromatic_ns
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer

RDLogger.DisableLog("rdApp.*")

INVALID_TYPES = [
    "all", "multiple", "exists", "par", "permut", "ring", "syntax", "valence", "arom",
]


def num_in_list(tokens: list[str]) -> list[str]:
    """Collects the distinct ring-closure digits present in a tokenized SMILES.

    :param tokens: SMILES tokens, e.g. as produced by
        :func:`~uncorrupt_smiles.utils.tokenizer.smi_tokenizer`.
    :return: Distinct ring-closure digit tokens (``"1"``-``"9"``) found in
        `tokens`, in no particular order.
    """
    symbols = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    nums = set()
    for a in tokens:
        if a in symbols:
            nums.add(a)
    return list(nums)


def exists_error(smi: str) -> str:
    """Introduces a ring-closure-digit-reuse error by removing or duplicating a
    ring-bond digit.

    :param smi: Valid SMILES string to corrupt.
    :return: The corrupted SMILES, unchanged if `smi` has no paired
        ring-closure digits to act on.
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    nums = num_in_list(tokens)
    if len(nums) > 0:
        avail_nums = list(nums)
        i = 0
        while i < len(avail_nums):
            num = random.choice(tuple(avail_nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            if (len(locs) % 2) == 0:
                index = random.choice(range(0, len(locs), 2))
                break
            else:
                avail_nums.remove(num)
                i += 1
        if "index" in locals():
            if random_value < 0.67:
                random_removal = random.random()
                if random_removal < 0.5:
                    tokens.pop(locs[index + 1])
                j = 1
                try:
                    while tokens[locs[index] + j] in ["(", "=", "#", "/", "-"]:
                        j += 1
                except IndexError:
                    pass
                tokens.insert(locs[index] + j + 1, num)
            else:
                if num == "1":
                    random_num = 2
                else:
                    random_num = int(num) + random.choice([-1, 1])
                tokens.insert(locs[index] + 1, str(random_num))
                random_loc = random.random()
                if random_loc < 0.5:
                    tokens.insert(locs[index + 1] + 1, str(random_num))
                else:
                    tokens.insert(locs[index + 1] + 2, str(random_num))

    return "".join(tokens[:])


def par_error(smi: str) -> str:
    """Introduces a parenthesis error: inserts, deletes, swaps, or flips a
    branch parenthesis.

    :param smi: Valid SMILES string to corrupt.
    :return: The corrupted SMILES, unchanged if `smi` has no parentheses to
        act on for the chosen mutation.
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    if random_value < 0.2:
        tokens.insert(random.randrange(1, len(tokens) + 1), random.choice(["(", ")"]))
    elif random_value < 0.4:
        index = [i for i, e in enumerate(tokens) if e in ("(", ")")]
        if index:
            tokens.pop(random.choice(index))
    elif random_value < 0.6:
        opening = [i for i, e in enumerate(tokens) if e == "("]
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        if opening and closing:
            tokens[random.choice(opening)] = ")"
            tokens[random.choice(closing)] = "("
    elif random_value < 0.8:
        opening = [i for i, e in enumerate(tokens) if e == "("]
        if opening:
            tokens[random.choice(opening)] = ")"
    else:
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        if closing:
            tokens[random.choice(closing)] = "("

    return "".join(tokens[:])


def permutation(smi: str, vocab: list[str]) -> tuple[str, list[str]]:
    """Applies a random token-level mutation (delete, insert, replace, or a
    short run thereof), drawing replacement tokens from `vocab`.

    :param smi: Valid SMILES string to corrupt.
    :param vocab: Candidate tokens to insert or substitute in.
    :return: A 2-tuple of the mutated SMILES and its token list; the token
        list is meant to be folded back into the caller's vocabulary pool.
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    if random_value < 0.17:
        tokens.pop(random.randrange(0, len(tokens)))
    elif random_value < 0.33:
        tokens.insert(random.randrange(0, len(tokens) + 1), random.choice(vocab))
    elif random_value < 0.5:
        tokens[random.randrange(0, len(tokens))] = random.choice(vocab)
        while smi == "".join(tokens[:]):
            tokens[random.randrange(0, len(tokens))] = random.choice(vocab)
    elif random_value < 0.66:
        i = 0
        try:
            start = random.randrange(0, len(tokens) - 1)
            stop = random.randrange(2, int(round(len(tokens) / 4) + 1))
            while i in range(0, stop) and i + start <= len(tokens):
                tokens.pop(start)
                i = i + 1
        except ValueError:
            pass
    elif random_value < 0.83:
        location = random.randrange(0, len(tokens) + 1)
        try:
            stop = random.randrange(2, int(round(len(tokens) / 4) + 1))
            for _ in range(stop):
                tokens.insert(location, random.choice(vocab))
        except ValueError:
            pass
    else:
        try:
            start = random.randrange(0, len(tokens) - 1)
            stop = start + random.randrange(2, int(round(len(tokens) / 4) + 1))
            while start in range(start, stop) and start < len(tokens):
                tokens[start] = random.choice(vocab)
                start = start + 1
        except ValueError:
            pass

    return "".join(tokens[:]), tokens


def ring_error(smi: str) -> str:
    """Introduces an unclosed or duplicated ring-closure-symbol error.

    :param smi: Valid SMILES string to corrupt.
    :return: The corrupted SMILES, unchanged if `smi` has no ring-closure
        digits to act on.
    """
    tokens = smi_tokenizer(smi)
    nums = num_in_list(tokens)
    if len(nums) > 0:
        random_value = random.random()
        if random_value < 0.5:
            num = random.choice(tuple(nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            loc = random.choice(locs)
            tokens.pop(loc)
            random_rep = random.random()
            if random_rep < 0.25:
                if num == "1":
                    tokens.insert(loc, "2")
                else:
                    tokens.insert(loc, str(int(num) + random.choice([-1, 1])))
            elif random_rep < 0.5:
                nums.append(str(len(nums) + 1))
                tokens.insert(loc, random.choice(nums))
                while smi == "".join(tokens[:]):
                    tokens.insert(loc, random.choice(nums))
        elif random_value < 0.75:
            nums.append(str(len(nums) + 1))
            loc = random.randrange(0, len(tokens))
            tokens[loc] = random.choice(nums)
        else:
            num = random.choice(tuple(nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            index = random.choice(range(0, len(locs), 2))
            tokens.insert(locs[index], num)

    return "".join(tokens[:])


def syntax_error(smi: str) -> str:
    """Introduces invalid bond/parenthesis placement or a malformed bracket
    pattern.

    :param smi: Valid SMILES string to corrupt.
    :return: The corrupted SMILES.
    """
    tokens = smi_tokenizer(smi)
    syn_sym = ["=", "#", "-", "(", ")"]
    random_value = random.random()
    nums: list[str] = []
    if random_value < 0.1:
        tokens.insert(0, random.choice(syn_sym))
    elif random_value < 0.2:
        tokens.insert(len(tokens), random.choice(syn_sym[:4]))
    elif random_value < 0.3:
        locs = [i for i, e in enumerate(tokens) if e in syn_sym[:3]]
        try:
            tokens.insert(random.choice(locs) + random.randint(0, 1), random.choice(syn_sym[:3]))
        except BaseException:
            pass
    elif random_value < 0.4:
        locs = [i for i, e in enumerate(tokens) if e == "("]
        try:
            tokens.insert(random.choice(locs), random.choice(syn_sym[:3]))
        except BaseException:
            pass
    elif random_value < 0.6:
        locs = [i for i, e in enumerate(tokens) if e == "("]
        if locs:
            loc = random.choice(locs)
            tokens.insert(loc, "(")
            random_deletion = random.random()
            if random_deletion < 0.5:
                try:
                    locs2 = [i for i in locs if i > loc]
                    tokens.pop(random.choice(locs2) + 1)
                except BaseException:
                    pass
    else:
        nums = num_in_list(tokens)

    if 0.6 <= random_value < 0.8:
        random_choice = random.random()
        try:
            ring_index = tokens.index("1")
        except ValueError:
            ring_index = 100
        if random_choice < 0.5 and ring_index < 5:
            tokens = tokens[ring_index:]
        else:
            pattern = r"\([^\)\(\\\/]{1,3}[1-9]"
            matches = re.findall(pattern, smi)
            if matches:
                match = f"\\{random.choice(matches)}"
                replace = f"({match[-1]}"
                try:
                    tokens = re.sub(match, replace, smi, count=1)
                except BaseException:
                    pass
    elif 0.8 < random_value < 1:
        opening = [i for i, e in enumerate(tokens) if e == "("]
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        random_insertion = random.random()
        i = 0
        if opening:
            if len(opening) == len(closing):
                while i < 5:
                    index = random.randint(0, len(opening) - 1)
                    i = +1
                    if index + 1 == len(opening):
                        tokens = tokens[:opening[index] + 1] + tokens[closing[index]:]
                    elif closing[index] < opening[index + 1]:
                        tokens = tokens[:opening[index] + 1] + tokens[closing[index]:]
                    elif index + 2 == len(opening):
                        tokens = tokens[:opening[index] + 1] + tokens[closing[index + 1]:]
                    elif closing[index] > opening[index + 1] and closing[index + 1] < opening[index + 2]:
                        tokens = tokens[:opening[index] + 1] + tokens[closing[index + 1]:]
                    else:
                        continue
                    if random_insertion < 0.5:
                        tokens.insert(opening[index] + 1, str(random.choice(syn_sym[:3] + nums)))
                    break
    return "".join(tokens[:])


def valence_error(smiles: str, fragment: str) -> tuple[str, str]:
    """Over-bonds an atom, either by attaching `fragment` via a new bond or by
    bumping an existing bond's order.

    :param smiles: Valid SMILES string to corrupt.
    :param fragment: SMILES of a fragment that may be attached to `smiles`.
    :return: A 2-tuple ``(mutated_smiles, target_smiles)``; `target_smiles` is
        ``smiles + "." + fragment`` if a fragment was attached, otherwise the
        original `smiles`.
    """
    core = Chem.MolFromSmiles(smiles)
    corfrag = smiles
    random_value = random.random()
    if random_value < 0.5:
        frag = Chem.MolFromSmiles(fragment)
        combo = Chem.CombineMols(core, frag)
        edcombo = Chem.EditableMol(combo)
        smarts = "[A!h]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            new_corfrag = smiles + "." + fragment
            if Chem.MolFromSmiles(new_corfrag) is not None:
                corfrag = new_corfrag
            core_num = str(random.choice(match))
            core_num = int(re.sub(r"(\(|,\))", "", core_num))
            frag_num = core.GetNumAtoms() + random.randrange(0, frag.GetNumAtoms())
            bond_order = random.choice([
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
            ])
            edcombo.AddBond(core_num, frag_num, order=bond_order)
            back = edcombo.GetMol()
            try:
                smiles = Chem.MolToSmiles(back)
            except BaseException:
                pass
    else:
        smarts = "[A!h]-,=*"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            center_num, neigh_num = random.choice(match)
            bond = core.GetBondBetweenAtoms(center_num, neigh_num)
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond_order = random.choice([Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
                bond.SetBondType(bond_order)
            else:
                bond.SetBondType(Chem.BondType.TRIPLE)
            try:
                smiles = Chem.MolToSmiles(core)
            except BaseException:
                pass

    return smiles, corfrag


def arom_error(smiles: str, fragment: str, use_mol: bool = True) -> tuple[str, str]:
    """Corrupts aromaticity: bumps a bond order, attaches `fragment`, flips
    atom case, or swaps a ring atom.

    :param smiles: Valid SMILES string to corrupt.
    :param fragment: SMILES of a fragment that may be attached to `smiles`.
    :param use_mol: If ``True``, allow mutations that parse `smiles` into an
        RDKit molecule; if ``False``, restrict to token-level mutations.
    :return: A 2-tuple ``(mutated_smiles, target_smiles)``; `target_smiles` is
        ``smiles + "." + fragment`` if a fragment was attached, otherwise the
        original `smiles`.
    """
    if use_mol:
        random_value = random.random()
    else:
        random_value = random.uniform(0.5, 1)

    corfrag = smiles
    core = None
    tokens: list[str] = []
    if random_value < 0.5:
        core = Chem.MolFromSmiles(smiles)
    else:
        tokens = smi_tokenizer(smiles)

    if random_value < 0.16:
        smarts = "c-[Ah!H1]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            center_num, neigh_num = random.choice(match)
            bond = core.GetBondBetweenAtoms(center_num, neigh_num)
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
                try:
                    smiles = Chem.MolToSmiles(core)
                except BaseException:
                    pass
    elif random_value < 0.33:
        frag = Chem.MolFromSmiles(fragment)
        combo = Chem.CombineMols(core, frag)
        edcombo = Chem.EditableMol(combo)
        smarts = "[c!h,nD2]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            try:
                new_corfrag = smiles + "." + fragment
                if Chem.MolFromSmiles(new_corfrag) is not None:
                    corfrag = new_corfrag
                core_num = str(random.choice(match))
                core_num = int(re.sub(r"(\(|,\))", "", core_num))
                frag_num = core.GetNumAtoms()
                edcombo.AddBond(core_num, frag_num, order=Chem.rdchem.BondType.SINGLE)
                back = edcombo.GetMol()
                try:
                    smiles = Chem.MolToSmiles(back)
                except BaseException:
                    pass
            except IndexError:
                pass
    elif random_value < 0.5:
        mol = Chem.rdchem.Mol(core.ToBinary())
        mol = adjust_aromatic_ns(mol)
        try:
            smiles = Chem.MolToSmiles(mol)
        except BaseException:
            pass
    elif random_value < 0.6:
        index = [i for i, e in enumerate(tokens) if e in ["C", "N", "O", "S", "P", "B"]]
        if index:
            index = random.choice(index)
            tokens[index] = tokens[index].lower()
        smiles = "".join(tokens[:])
    elif random_value < 0.7:
        index_c = [i for i, e in enumerate(tokens) if e == "c"]
        index_n = [i for i, e in enumerate(tokens) if e == "n"]
        options = [
            "[nH]", ["n", "(", "C", ")"], "o", "s",
            ["c", "(", "=", "O", ")"], ["c", "(", "=", "N", ")"],
            "C", "N", ["c", "c"], ["c", "n"], ["n", "c"], ["n", "n"],
        ]
        index = random.choice([index_c, index_n])
        if index:
            index = random.choice(index)
            tokens.pop(index)
            tokens[index:index] = random.choice(options)
        smiles = "".join(tokens[:])
    elif random_value < 0.8:
        index = [i for i, e in enumerate(tokens) if e.islower()]
        if index:
            index = random.choice(index) + random.choice([0, 1])
            tokens.insert(index, random.choice(["c", "n"]))
        smiles = "".join(tokens[:])
    elif random_value < 0.9:
        index = [i for i, e in enumerate(tokens) if e in ["[nH]", "o", "s"]]
        if index:
            index = random.choice(index)
            tokens[index] = random.choice(["c", "n"])
        smiles = "".join(tokens[:])
    else:
        index = [i for i, e in enumerate(tokens) if e in ("c", "n")]
        if index:
            index = random.choice(index)
            tokens.pop(index)
        smiles = "".join(tokens[:])

    return smiles, corfrag


def introduce_error(
    smile: str, fragment: str, vocab: set[str], invalid_type: str = "all", num_errors: int = 1
) -> tuple[str | None, str]:
    """Repeatedly corrupts `smile` until RDKit can no longer parse it, or 20
    attempts are exhausted.

    :param smile: Valid SMILES string to corrupt.
    :param fragment: SMILES of a fragment that may be attached during a
        valence or aromaticity error.
    :param vocab: Replacement-token pool for :func:`permutation`; updated in
        place with newly seen tokens.
    :param invalid_type: Which error type(s) to apply; one of
        :data:`INVALID_TYPES`. ``"multiple"`` additionally applies
        `num_errors` - 1 further mutations once `smile` first becomes
        invalid.
    :param num_errors: Total number of mutations to apply when
        `invalid_type` is ``"multiple"``; ignored otherwise.
    :return: A 2-tuple ``(invalid_smile, target_smiles)``; `invalid_smile` is
        ``None`` if `smile` could not be made invalid within 20 attempts, and
        `target_smiles` is ``smile + "." + fragment`` if a fragment was
        attached, otherwise the original `smile`.
    """
    corfrag = smile
    i = 0

    while is_valid_smiles(smile) and i < 20:
        i += 1
        try:
            if invalid_type in ("all", "multiple"):
                random_value = random.choice(range(1, 8))
                if random_value == 1:
                    smile = exists_error(smile)
                elif random_value == 2:
                    smile = par_error(smile)
                elif random_value == 3:
                    smile, tokens = permutation(smile, list(vocab))
                    vocab.update(tokens)
                elif random_value == 4:
                    smile = ring_error(smile)
                elif random_value == 5:
                    smile = syntax_error(smile)
                elif random_value == 6:
                    smile, corfrag = valence_error(smile, fragment)
                elif random_value == 7:
                    smile, corfrag = arom_error(smile, fragment)
            elif invalid_type == "exists":
                smile = exists_error(smile)
            elif invalid_type == "par":
                smile = par_error(smile)
            elif invalid_type == "permut":
                smile, tokens = permutation(smile, list(vocab))
                vocab.update(tokens)
            elif invalid_type == "ring":
                smile = ring_error(smile)
            elif invalid_type == "syntax":
                smile = syntax_error(smile)
            elif invalid_type == "valence":
                smile, corfrag = valence_error(smile, fragment)
            elif invalid_type == "arom":
                smile, corfrag = arom_error(smile, fragment)
        except ValueError:
            pass

    if invalid_type == "multiple":
        for _ in range(num_errors - 1):
            original_smile = smile
            while smile == original_smile and len(smile) > 0:
                random_value = random.choice(range(1, 7))
                if random_value == 1:
                    smile = exists_error(smile)
                elif random_value == 2:
                    smile = par_error(smile)
                elif random_value == 3:
                    smile, tokens = permutation(smile, list(vocab))
                    vocab.update(tokens)
                elif random_value == 4:
                    smile = ring_error(smile)
                elif random_value == 5:
                    smile = syntax_error(smile)
                elif random_value == 6:
                    smile, _ = arom_error(smile, fragment, use_mol=False)

    if is_valid_smiles(smile):
        smile = None

    return smile, corfrag


def reservoir_sample_fragments(path: str, column: str, k: int, seed: int) -> list[str]:
    """Draws a fixed-size uniform random sample from a CSV column via one
    streaming pass (reservoir sampling).

    :param path: Path to the CSV file to sample from.
    :param column: Name of the column to sample.
    :param k: Sample (reservoir) size; memory use is bounded by `k`, not by
        the file's size.
    :param seed: Seed for the sampling RNG.
    :return: Up to `k` sampled values; order is not meaningful.
    """
    rng = random.Random(seed)
    pool: list[str] = []
    for i, value in enumerate(iter_csv_column(path, column)):
        if i < k:
            pool.append(value)
        else:
            j = rng.randint(0, i)
            if j < k:
                pool[j] = value
    return pool


def build_seed_vocab(smiles: Iterable[str], sample_size: int = 50) -> set[str]:
    """Tokenizes up to `sample_size` valid SMILES into an initial
    :func:`permutation` replacement-token vocabulary.

    Mirrors the original implementation's ``df.sample(50)`` seed step, but
    streamed from the start of `smiles` rather than a true random sample.

    :param smiles: SMILES strings to draw the seed sample from.
    :param sample_size: Maximum number of (valid) SMILES to tokenize.
    :return: The set of distinct tokens seen across the sampled SMILES.
    """
    vocab: set[str] = set()
    for i, smi in enumerate(smiles):
        if i >= sample_size:
            break
        if is_valid_smiles(smi):
            vocab.update(smi_tokenizer(smi))
    return vocab


def generate_errors(
    smiles: Iterable[str],
    fragment_pool: list[str],
    seed: int,
    invalid_type: str = "all",
    num_errors: int = 1,
    vocab: set[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Streams ``(target_smiles, invalid_smiles)`` pairs, one per valid input
    SMILES.

    Input SMILES that RDKit cannot parse are skipped rather than passed
    through. `vocab`, if given, seeds :func:`permutation`'s replacement-token
    pool and grows online exactly as :func:`introduce_error` does.

    :param smiles: Input SMILES strings to corrupt.
    :param fragment_pool: Candidate fragments for valence/aromaticity errors;
        one is chosen at random per input SMILES.
    :param seed: Seed for the module-level random generator (via
        :func:`random.seed`).
    :param invalid_type: Which error type(s) to apply; one of
        :data:`INVALID_TYPES`.
    :param num_errors: Number of mutations to apply per SMILES when
        `invalid_type` is ``"multiple"``.
    :param vocab: Initial replacement-token pool for :func:`permutation`; a
        fresh empty set is used if ``None``.
    :raises ValueError: If `invalid_type` is not one of :data:`INVALID_TYPES`.
    :return: Iterator of ``(target_smiles, invalid_smiles)`` pairs; SMILES
        that :func:`introduce_error` fails to invalidate are skipped.
    """
    if invalid_type not in INVALID_TYPES:
        raise ValueError(f"invalid_type must be one of {INVALID_TYPES}, got {invalid_type!r}")
    random.seed(seed)
    vocab = set(vocab) if vocab is not None else set()
    for smi in smiles:
        if not is_valid_smiles(smi):
            continue
        fragment = random.choice(fragment_pool)
        invalid, target = introduce_error(smi, fragment, vocab, invalid_type, num_errors)
        # introduce_error can degenerate to an empty string via repeated deletions; that's
        # not a useful (or even a real) invalid SMILES, so treat it like a failed attempt.
        if not invalid:
            continue
        yield target, invalid


def write_errors_split(
    input_csv: str,
    smiles_col: str,
    fragment_csv: str,
    fragment_col: str,
    train_csv: str,
    dev_csv: str,
    seed: int,
    invalid_type: str = "all",
    num_errors: int = 1,
    fragment_pool_size: int = 20_000,
    length_threshold: int = 200,
    frac_train: float = 0.9,
    vocab_seed_size: int = 50,
) -> None:
    """Streams `input_csv` through :func:`generate_errors`, filters by token
    length, and writes a deterministic per-row train/dev split.

    Fuses what would otherwise be three full-dataframe passes (error
    generation, length filtering, train/test split) into two streaming passes
    over the input (one lightweight vocab-seed peek, one full pass), with no
    dataset ever fully materialized in memory.

    :param input_csv: Path to the CSV of valid SMILES to corrupt.
    :param smiles_col: Name of the SMILES column in `input_csv`.
    :param fragment_csv: Path to the CSV of candidate fragments.
    :param fragment_col: Name of the fragment column in `fragment_csv`.
    :param train_csv: Output path for the training split.
    :param dev_csv: Output path for the dev split.
    :param seed: Seed for fragment sampling, error generation, and the
        train/dev split.
    :param invalid_type: Which error type(s) to apply; one of
        :data:`INVALID_TYPES`.
    :param num_errors: Number of mutations to apply per SMILES when
        `invalid_type` is ``"multiple"``.
    :param fragment_pool_size: Number of fragments to reservoir-sample from
        `fragment_csv`.
    :param length_threshold: Maximum token length (for either the target or
        the invalid SMILES) allowed in the output; longer pairs are dropped.
    :param frac_train: Fraction of (post-filter) pairs routed to
        `train_csv`; the remainder goes to `dev_csv`.
    :param vocab_seed_size: Number of SMILES used to seed the initial
        :func:`permutation` vocabulary.
    :return: None
    """
    fragment_pool = reservoir_sample_fragments(fragment_csv, fragment_col, fragment_pool_size, seed)
    seed_vocab = build_seed_vocab(iter_csv_column(input_csv, smiles_col), vocab_seed_size)
    pairs = generate_errors(
        iter_csv_column(input_csv, smiles_col), fragment_pool, seed,
        invalid_type=invalid_type, num_errors=num_errors, vocab=seed_vocab,
    )

    split_rng = random.Random(seed)
    with open(train_csv, "w", newline="") as f_train, open(dev_csv, "w", newline="") as f_dev:
        writer_train = csv.writer(f_train)
        writer_dev = csv.writer(f_dev)
        writer_train.writerow(["STD_SMILES", "ERROR"])
        writer_dev.writerow(["STD_SMILES", "ERROR"])
        for target, invalid in pairs:
            if len(smi_tokenizer(target)) > length_threshold or len(smi_tokenizer(invalid)) > length_threshold:
                continue
            writer = writer_train if split_rng.random() < frac_train else writer_dev
            writer.writerow([target, invalid])
