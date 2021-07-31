#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:46:39 2021

@author: akshat
"""
import random
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import dask.dataframe as dd
import selfies
from selfies import encoder, decoder
import multiprocessing

# Updated SELFIES constraints: 
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints


def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def obtain_path(starting_smile, target_smile, filter_path=False): 
    starting_selfie = encoder(starting_smile)
    target_selfie   = encoder(target_smile)
    
    starting_selfie_chars = get_selfie_chars(starting_selfie)
    target_selfie_chars   = get_selfie_chars(target_selfie)
    
    # Pad the smaller string
    if len(starting_selfie_chars) < len(target_selfie_chars): 
        for _ in range(len(target_selfie_chars)-len(starting_selfie_chars)):
            starting_selfie_chars.append(' ')
    else: 
        for _ in range(len(starting_selfie_chars)-len(target_selfie_chars)):
            target_selfie_chars.append(' ')
    
    indices_diff = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != target_selfie_chars[i]]
    path         = {}
    path[0]  = starting_selfie_chars
    
    for iter_ in range(len(indices_diff)): 
        idx = np.random.choice(indices_diff, 1)[0] # Index to be operated on
        indices_diff.remove(idx)                   # Remove that index
        
        # Select the last member of path: 
        path_member = path[iter_].copy()
        
        # Mutate that character to the correct value: 
        path_member[idx] = target_selfie_chars[idx]
        path[iter_+1] = path_member.copy()
    
    # Collapse path to make them into SELFIE strings
    paths_selfies = []
    for i in range(len(path)):
        selfie_str = ''.join(x for x in path[i])
        paths_selfies.append(selfie_str.replace(' ', ''))
        
    if paths_selfies[-1] != target_selfie: 
        raise Exception("Unable to discover target structure!")
    
    path_smiles         = [decoder(x) for x in paths_selfies]

    return path_smiles




def perform_crossover(comb_smi, num_random_samples): 
    
    smi_a, smi_b = comb_smi.split('xxx')
    mol_a, mol_b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    Chem.Kekulize(mol_a)
    Chem.Kekulize(mol_b)
    
    randomized_smile_orderings_a = []
    for _ in range(num_random_samples): 
        randomized_smile_orderings_a.append(rdkit.Chem.MolToSmiles(mol_a, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))
    
    randomized_smile_orderings_b = []
    for _ in range(num_random_samples): 
        randomized_smile_orderings_b.append(rdkit.Chem.MolToSmiles(mol_b, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True))

    collect_smiles = []
    for smi_1 in randomized_smile_orderings_a: 
        for smi_2 in randomized_smile_orderings_b: 
            for item in obtain_path(smi_1, smi_2): 
                collect_smiles.append(item)
                
    collect_smiles_canon = []
    for item in collect_smiles: 
        try: 
            smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(item, sanitize=True), isomericSmiles=False, canonical=True)
            if len(smi_canon) <= 81: # Size restriction! 
                collect_smiles_canon.append(smi_canon)       
        except: 
            continue 
            
    collect_smiles_canon = list(set(collect_smiles_canon))

    return collect_smiles_canon

def get_fp_scores(smiles_back, target_smi): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores


def get_joint_sim(all_smiles, starting_smile, target_smile): 
    scores_start  = get_fp_scores(all_smiles, starting_smile)   # similarity to target
    scores_target = get_fp_scores(all_smiles, target_smile)     # similarity to starting structure
    data          = np.array([scores_target, scores_start])
    
    avg_score     = np.average(data, axis=0)
    better_score  = avg_score - (np.abs(data[0] - data[1]))   
    better_score  = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)
    
    return better_score



def crossover_smiles(smiles_join): 
    
    map_ = {}
    for i, item in enumerate(smiles_join):
        if i % 10 == 0: 
            print('Cross: {}/{}'.format(i, len(smiles_join)))
        map_[item] = perform_crossover(item, num_random_samples=2)

    map_ordered = {}
    for key_ in map_: 
        med_all      = map_[key_]
        smi_1, smi_2 = key_.split('xxx')
        joint_sim    = get_joint_sim(med_all, smi_1, smi_2)
        
        joint_sim_ord = np.argsort(joint_sim)
        joint_sim_ord = joint_sim_ord[::-1]
        
        med_all_ord = [med_all[i] for i in joint_sim_ord]
        map_ordered[key_] = med_all_ord

    return map_ordered


