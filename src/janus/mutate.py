#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 12:15:57 2021

@author: akshat
"""
from typing import Dict
import random
import multiprocessing

import rdkit
from rdkit import Chem

import selfies 
from selfies import encoder, decoder

from .utils import get_selfies_chars

def mutate_sf(sf_chars, alphabet, num_sample_frags, base_alphabet = None):
    """
    Given a list of SELFIES alphabets, make random changes to the molecule using 
    alphabet. Opertations to molecules are character replacements, additions and deletions. 

    Parameters
    ----------
    sf_chars : (list of string alphabets)
        List of string alphabets for a SELFIE string.
    alphabet : (list of SELFIE strings)
        New SELFIES characters are added here and sampled.
    num_sample_frags: (int)
        Number of randomly sampled SELFIE strings.
    base_alphabet: (list of SELFIE strings)
        Main alphabet that will be appended with the introduced characters above.
        If none, use the semantic robust alphabet.

    Returns
    -------
    Mutated SELFIE string.

    """
    if base_alphabet is None:
        base_alphabet = list(selfies.get_semantic_robust_alphabet())
    random_char_idx = random.choice(range(len(sf_chars)))
    choices_ls = [1, 2, 3]  # 1 = replacement; 2 = addition; 3=delete
    mutn_choice = choices_ls[
        random.choice(range(len(choices_ls)))
    ]  # Which mutation to do:

    if alphabet != []:
        alphabet = random.sample(alphabet, num_sample_frags) + base_alphabet
    else:
        alphabet = base_alphabet

    # Mutate character:
    if mutn_choice == 1:
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf = (
            sf_chars[0:random_char_idx]
            + [random_char]
            + sf_chars[random_char_idx + 1 :]
        )

    # add character:
    elif mutn_choice == 2:
        random_char = alphabet[random.choice(range(len(alphabet)))]
        change_sf = (
            sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx:]
        )

    # delete character:
    elif mutn_choice == 3:
        if len(sf_chars) != 1:
            change_sf = sf_chars[0:random_char_idx] + sf_chars[random_char_idx + 1 :]
        else:
            change_sf = sf_chars

    return "".join(x for x in change_sf)


def mutate_smiles(
    smile, alphabet, num_random_samples, num_mutations, num_sample_frags, base_alphabet = None
):
    """
    Given an input smile, perform mutations to the strucutre using provided SELFIE
    alphabet list. 'num_random_samples' number of different SMILES orientations are 
    considered & total 'num_mutations' are performed. 

    Parameters
    ----------
    smile : (str)
        Valid SMILES string.
    alphabet : (list of str)
        list of SELFIE strings.
    num_random_samples : (int)
        Number of different SMILES orientations to be formed for the input smile.
    num_mutations : TYPE
        Number of mutations to perform on each of different orientations SMILES.
    num_sample_frags: (int)
        Number of randomly sampled SELFIE strings.

    Returns
    -------
    mutated_smiles_canon : (list of strings)
        List of unique molecules produced from mutations.
    """
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol)

    # Obtain randomized orderings of the SMILES:
    randomized_smile_orderings = []
    for _ in range(num_random_samples):
        randomized_smile_orderings.append(
            rdkit.Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
        )

    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    selfies_ls_chars = [get_selfies_chars(selfie) for selfie in selfies_ls]

    # Obtain the mutated selfies
    mutated_sf = []
    for sf_chars in selfies_ls_chars:

        for i in range(num_mutations):
            if i == 0:
                mutated_sf.append(mutate_sf(sf_chars, alphabet, num_sample_frags, base_alphabet))
            else:
                mutated_sf.append(
                    mutate_sf(
                        get_selfies_chars(mutated_sf[-1]), alphabet, num_sample_frags, base_alphabet
                    )
                )

    mutated_smiles = [decoder(x) for x in mutated_sf]
    mutated_smiles_canon = []
    for item in mutated_smiles:
        try:
            smi_canon = Chem.MolToSmiles(
                Chem.MolFromSmiles(item, sanitize=True),
                isomericSmiles=False,
                canonical=True,
            )
            if smi_canon != "": 
                mutated_smiles_canon.append(smi_canon)
        except:
            continue

    mutated_smiles_canon = list(set(mutated_smiles_canon))
    return mutated_smiles_canon



if __name__ == "__main__":
    molecules_here = [
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCCCC",
        "CS",
        "CSSS",
        "CSSSSS",
        "CF",
        "CI",
        "CBr",
        "CSSSSSSSSSSSS",
        "CSSSSSSSSSC",
        "CSSSSCCSSSC",
        "CSSSSSSSSSF",
        "SSSSSC",
    ]
    A = get_mutated_smiles(
        molecules_here, alphabet=["[C]"] * 500, num_sample_frags=200, space="Explore"
    )

