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
from uncorrupt_smiles.utils.sanifix import AdjustAromaticNs
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer

RDLogger.DisableLog("rdApp.*")

INVALID_TYPES = [
    "all", "multiple", "exists", "par", "permut", "ring", "syntax", "valence", "arom",
]


def num_in_list(tokens: list[str]) -> list[str]:
    """Ring-closure digits (1-9) present in a tokenized SMILES."""
    symbols = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    nums = set()
    for a in tokens:
        if a in symbols:
            nums.add(a)
    return list(nums)


def exists_error(smi: str) -> str:
    """Ring-closure-digit-already-in-use style error: remove or duplicate a ring bond number."""
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
    """Parenthesis insert/delete/swap/flip error."""
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
    """Random token delete/insert/replace, or a short stretch thereof, drawing replacement
    tokens from `vocab`. Returns (mutated_smiles, tokens) - `tokens` feeds back into the
    caller's vocab set."""
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
    """Unclosed/duplicated ring-closure symbol error."""
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
    """Invalid bond/parenthesis placement, malformed bracket patterns."""
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
    """Over-bonds an atom (add fragment or bump bond order). Returns (mutated_smiles,
    target_smiles) - target may be `smiles + "." + fragment` if a fragment was attached."""
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
    """Aromaticity corruption: bond-order bump, fragment addition, case-flipping, ring-atom
    swaps. Returns (mutated_smiles, target_smiles)."""
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
        mol = AdjustAromaticNs(mol)
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
    """Corrupts `smile` until RDKit can no longer parse it (or 20 tries are exhausted).
    Returns (invalid_smile_or_None, target_smiles) - target_smiles is `smile` unless a
    valence/aromaticity error attached a fragment, in which case it's `smile.fragment`."""
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
    """One streaming pass building a fixed-size (k) sample pool via reservoir sampling.
    Memory is bounded by k, not by the fragment file's size."""
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
    """Tokenizes up to `sample_size` SMILES into the initial permutation() vocabulary,
    mirroring the original's `df.sample(50)` seed step (but streamed, from the start of
    the iterable, rather than a true random sample)."""
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
    """Streams (target_smiles, invalid_smiles) pairs, one per input SMILES. Any input SMILES
    RDKit can't parse is skipped rather than passed through. `vocab` seeds permutation()'s
    replacement-token pool and grows online exactly as introduce_error already does."""
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
    """Streams input_csv -> generate_errors() -> per-row token-length filter -> deterministic
    per-row train/dev split -> incremental writes to train_csv/dev_csv. Fuses what used to be
    three full-dataframe passes (error generation, length filter, sklearn train_test_split)
    into two streaming passes over the input (one lightweight vocab-seed peek, one full pass)
    with no dataset ever fully materialized."""
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
