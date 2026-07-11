import torch
import os

import pandas as pd
import numpy as np

import random

import pickle

from rdkit import Chem

from uncorrupt_smiles.invalidSMILES import generate_errors, reservoir_sample_fragments
from uncorrupt_smiles.preprocess import standardize_smiles
from uncorrupt_smiles.transformer import Seq2Seq

if __name__ == "__main__":
    # set random seed, used for error generation & initiation transformer
    SEED = 42
    random.seed(SEED)

    name = 'selective_ki'
    errors_per_molecule = 1000
    if not os.path.exists(f"data/explore"):
        os.makedirs(f"data/explore")
    error_source = "data/explore/%s_with_%s_errors_index.csv" % (
        name, errors_per_molecule)

    folder_raw = "rawdata/"
    folder_out = "data/"
    invalid_type = 'multiple'
    num_errors = 12
    threshold = 200
    data_source = f"PAPYRUS_{threshold}"
    # point this at a checkpoint written by `uncorrupt-smiles train --checkpoint-out ...`
    checkpoint_path = f"{folder_out}performance/transformer_{invalid_type}_{num_errors}_{data_source}.pkg"

    # introduce = True

    standardize = False
    if standardize:
        df = pd.read_csv('%s%s.csv' % (folder_out, name), usecols=['SMILES']).dropna()
        df["STD_SMILES"] = df.apply(
            lambda row: standardize_smiles(row["SMILES"]),
            axis=1).dropna()
        df = df.drop(columns=['SMILES'])
        df.to_csv('%s%s.csv' % (folder_out, name), index=None)
    else:
        df = pd.read_csv('%s%s.csv' % (folder_out, name),
                         usecols=['STD_SMILES']).dropna()

    introduce = False
    if introduce:
        fragment_pool = reservoir_sample_fragments(
            f"{folder_raw}gbd_8.csv", "FRAGMENT", 20_000, SEED)

        # repeat each molecule errors_per_molecule times so introduce_error's randomness
        # produces that many independent corruptions per molecule
        smiles = list(df['STD_SMILES'].values)
        repeated = smiles * errors_per_molecule
        pairs = list(generate_errors(repeated, fragment_pool, SEED, invalid_type="all", num_errors=1))

        df = pd.DataFrame(pairs, columns=["SMILES_TARGET", "SMILES"])
        df["ORIGINAL_SMILES"] = df["SMILES_TARGET"]
        df = df.drop_duplicates(subset=['SMILES'])
        df.to_csv(error_source)

    correct = False
    if correct:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, src_vocab, trg_vocab = Seq2Seq.load_checkpoint(checkpoint_path, device)
        fixed_path = f"data/explore/{error_source.split('/')[2].split('.')[0]}_fixed.csv"
        model.fix_smiles_csv(error_source, "SMILES", fixed_path, src_vocab, trg_vocab)

    df_new = pd.read_csv(f"data/explore/{error_source.split('/')[2].split('.')[0]}_fixed.csv")
    df_new = df_new[df_new["FIXED"].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    df_new["STD_SMILES"] = df_new["FIXED"].apply(standardize_smiles)
    df_new = df_new.dropna(subset=["STD_SMILES"]).drop_duplicates(subset=["STD_SMILES"])
