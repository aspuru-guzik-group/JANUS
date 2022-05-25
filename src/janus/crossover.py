#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:39:03 2021

@author: akshat
"""
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import selfies 
from selfies import encoder, decoder
import multiprocessing

from rdkit import RDLogger

from .utils import get_selfies_chars

RDLogger.DisableLog("rdApp.*")

def get_fp_scores(smiles_back, target_smi):
    """
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 

    Parameters
    ----------
    smiles_back : (list of str)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 

    Returns
    -------
    smiles_back_scores : (list of floats)
        List of figerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = Chem.MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = Chem.MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_joint_sim(all_smiles, starting_smile, target_smile):
    """
    Get joint similarity values for all smiles in all_smiles, calculated with 
    refernce to starting_smile & target_smile. 

    Parameters
    ----------
    all_smiles : (list of string)
        List of SMILE strings.
    starting_smile : (str)
        Input smiles string.
    target_smile : (str)
        Input smiles string.

    Returns
    -------
    better_score : (list of floats)
        List of joint similarity scores for all smiles in all_smiles.
    """
    scores_start = get_fp_scores(all_smiles, starting_smile)  # similarity to target
    scores_target = get_fp_scores(
        all_smiles, target_smile
    )  # similarity to starting structure
    data = np.array([scores_target, scores_start])

    avg_score = np.average(data, axis=0)
    better_score = avg_score - (np.abs(data[0] - data[1]))
    better_score = (
        ((1 / 9) * better_score ** 3)
        - ((7 / 9) * better_score ** 2)
        + ((19 / 12) * better_score)
    )

    return better_score


def obtain_path(starting_smile, target_smile):
    """
    Create a single path between molecules starting_smile and target_smile. 

    Parameters
    ----------
    starting_smile : (str)
        Valid SMILES string.
    target_smile : (str)
        Valid SMILES string.

    Returns
    -------
    path_smiles : (list of str)
        List of all smiles strings encountered while creating a path.
    """
    starting_selfie = encoder(starting_smile)
    target_selfie = encoder(target_smile)

    starting_selfie_chars = get_selfies_chars(starting_selfie)
    target_selfie_char = get_selfies_chars(target_selfie)

    # Pad the smaller string
    if len(starting_selfie_chars) < len(target_selfie_char):
        for _ in range(len(target_selfie_char) - len(starting_selfie_chars)):
            starting_selfie_chars.append(" ")
    else:
        for _ in range(len(starting_selfie_chars) - len(target_selfie_char)):
            target_selfie_char.append(" ")

    indices_diff = [
        i
        for i in range(len(starting_selfie_chars))
        if starting_selfie_chars[i] != target_selfie_char[i]
    ]
    path = {}
    path[0] = starting_selfie_chars

    for iter_ in range(len(indices_diff)):
        idx = np.random.choice(indices_diff, 1)[0]  # Index to be operated on
        indices_diff.remove(idx)  # Remove that index

        # Select the last member of path:
        path_member = path[iter_].copy()

        # Mutate that character to the correct value:
        path_member[idx] = target_selfie_char[idx]
        path[iter_ + 1] = path_member.copy()

    # Collapse path to make them into SELFIE strings
    paths_selfies = []
    for i in range(len(path)):
        selfie_str = "".join(x for x in path[i])
        paths_selfies.append(selfie_str.replace(" ", ""))

    if paths_selfies[-1] != target_selfie:
        raise Exception("Unable to discover target structure!")
    path_smiles = [decoder(x) for x in paths_selfies]

    return path_smiles


def perform_crossover(comb_smi, num_random_samples):
    """
    Create multiple paths between SMILES in comb_smi to obtain median molecules, 
    representing the crossover structure. 

    Parameters
    ----------
    comb_smi : (str)
        Two smiles string concatenated using xxx (example: CCCCCCxxxSSS).
    num_random_samples : (int)
        Number of different smiles orientations to consider while forming paths. 

    Returns
    -------
    collect_smiles_canon : (list of SMILES)
        List of all potential unique median molecules enoucntered during path formation.
    """
    smi_a, smi_b = comb_smi.split("xxx")
    mol_a, mol_b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    Chem.Kekulize(mol_a)
    Chem.Kekulize(mol_b)

    randomized_smile_orderings_a = []
    for _ in range(num_random_samples):
        randomized_smile_orderings_a.append(
            rdkit.Chem.MolToSmiles(
                mol_a,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
        )

    randomized_smile_orderings_b = []
    for _ in range(num_random_samples):
        randomized_smile_orderings_b.append(
            rdkit.Chem.MolToSmiles(
                mol_b,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
        )

    collect_smiles = []
    for smi_1 in randomized_smile_orderings_a:
        for smi_2 in randomized_smile_orderings_b:
            for item in obtain_path(smi_1, smi_2):
                collect_smiles.append(item)

    collect_smiles_canon = []
    for item in collect_smiles:
        try:
            smi_canon = Chem.MolToSmiles(
                Chem.MolFromSmiles(item, sanitize=True),
                isomericSmiles=False,
                canonical=True,
            )
            if  smi_canon != "":  # Size restriction!
                collect_smiles_canon.append(smi_canon)
        except:
            continue

    collect_smiles_canon = list(set(collect_smiles_canon))

    return collect_smiles_canon


def crossover_smiles(smiles_join, crossover_num_random_samples):
    """
    Return a list of smiles (crossover molecules) that are ordered (highest to lowest)
    by joint similarity scores. 

    Parameters
    ----------
    smiles_join : (str)
        Two smiles string concatenated using xxx (example: CCCCCCxxxSSS).

    Returns
    -------
    med_all_ord : (list of SMILES)
        List of crossover molecules that are ordered (highest to lowest)
        by joint similarity scores.
    """
    map_ = {}

    map_[smiles_join] = perform_crossover(
        smiles_join, num_random_samples=crossover_num_random_samples
    )

    # map_ordered = {}
    for key_ in map_:
        med_all = map_[key_]
        smi_1, smi_2 = key_.split("xxx")
        joint_sim = get_joint_sim(med_all, smi_1, smi_2)

        joint_sim_ord = np.argsort(joint_sim)
        joint_sim_ord = joint_sim_ord[::-1]

        med_all_ord = [med_all[i] for i in joint_sim_ord]

    return med_all_ord
    